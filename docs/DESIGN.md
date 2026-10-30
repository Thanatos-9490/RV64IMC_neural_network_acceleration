# DESIGN - model choice and ISA extension

## 1. Model: ResNet-18, W8A8

| Item | Value |
|---|---|
| Model | `models/resnet18.tflite` (FP32 source) -> `models/resnet18_w8a8.tflite` |
| Size | 11.68 M weights, 1.814 GMAC (20 CONV_2D + 1 FULLY_CONNECTED = 100% of MACs) |
| Precision | **W8A8** (INT8 weights, INT8 activations, INT32 accumulate), per-channel weight scales |
| Quantization | TF 2.15 post-training quantization, 50 calibration images (`scripts/quantize_model.py`) |

Accuracy (top-1, 1000-way, 500 Imagenette-val images; sampling error about +-4 pt):

| Model | FP32 | W8A8 | W8A16 |
|---|---:|---:|---:|
| ResNet-18 | 64.4% | **65.8%** | 65.2% |


## 2. Lowering to GEMM

Every layer becomes `out[M][N] = requant(bias'[n] + sum_k A[m][k] * W[n][k])`
(`scripts/layer_build.py`):

- CONV_2D: im2col (A = patches, tap order ky,kx,c), SAME padding filled with the
  input zero-point.  FC: M=1.
- The input zero-point is folded into the bias: `bias' = bias - z_in * sum_k W[n][k]`
  (exact in integers), so the inner loop is a pure signed INT8 dot product.
- K is zero-padded to `Kp` (multiple of 8) so rows are 8-byte aligned and
  `qmac8` never needs a tail loop.  Padding is zeros, so the result is unchanged.

The numpy implementation of this lowering equals the TFLite interpreter
(`BUILTIN_REF` kernels) byte-for-byte on all 21 layers (`gen/layers.json`,
`numpy_matches_tflite`), and the bare-metal kernels equal it on Spike.

## 3. Instructions (custom-0 = 0x0b R-type; custom-1 = 0x2b I-type)

| | funct7 | funct3 | encoding match / mask |
|---|---|---|---|
| `qmac8 rd, rs1, rs2` | 0x01 | 0 | `0x0200000b` / `0xfe00707f` |
| `qreq  rd, rs1, rs2` | 0x02 | 0 | `0x0400000b` / `0xfe00707f` |
| `ldpi  rd, imm(rs1)` (X2, I-type, opcode custom-1 = 0x2b) | - | 0 | `0x0000002b` / `0x0000707f` |

### qmac8 - 8 x INT8 multiply-accumulate

```
x[rd] = sext32( (int32)x[rd] + sum_{i=0..7} int8(rs1[8i+7:8i]) * int8(rs2[8i+7:8i]) )
```

`rd` is both accumulator input and output.  **Accumulator width: 32 bit**
(wraps modulo 2^32, same as the TFLite INT32 accumulator).

Range check for W8A8 (measured on the exported layers, `gen/`):

- Dot-product part: worst-case bound `K*128*128` = 7.5e7 for K=4608; the largest
  value actually observed is 2.9e6 (`op29`), so there is >700x headroom.
- The accumulator is seeded with the folded bias `bias'`.  On `op02` (conv1)
  `|bias'|` reaches 1.07e9 (the input zero-point times the weight sum is large),
  so `|acc|` peaks at 1.074e9, i.e. 50% of 2^31.  It does not overflow, but
  conv1 is the layer with the least margin, and it is a property of the bias,
  not of the MAC.

### qreq - requantization, bit-exact with TFLite

```
rs1       = INT32 accumulator
rs2[31:0] = multiplier (Q31)      rs2[39:32] = shift (int8; >0 left, <=0 right)
rs2[47:40]= output zero point     rs2[55:48] = act_min   rs2[63:56] = act_max
x[rd]     = clamp( RoundingDivideByPOT( SaturatingRoundingDoublingHighMul(
                      acc << max(shift,0), multiplier ), max(-shift,0) ) + zp,
                   act_min, act_max )
```

This is TFLite's `MultiplyByQuantizedMultiplier` (gemmlowp semantics: two
roundings, ties away from zero), plus the fused-activation clamp.  A single
canonical C definition (`src/tfl_requant.h`) is used by the Spike handler, the
host model and the software-requant kernels, and is checked against an
independent numpy implementation (`scripts/tflq.py`, 4824 golden vectors) and
against the TFLite interpreter on real layers.

An earlier draft rounded once; it disagreed with TFLite on 0.11% of
non-saturated outputs, so it was replaced.

### Design rationale and trade-offs (qmac8 / qreq)

| Decision | Choice | Why | Cost / alternative |
|---|---|---|---|
| Operand packing | 8 x int8 in one 64-bit register, lane i = bits 8i+7:8i | RV64 `ld` loads 8 int8 as-is: no pack/unpack instructions | needs K padded to a multiple of 8 (zeros, no effect on the result) |
| Accumulator | rd is read+write, 32 bit | same width as the TFLite INT32 accumulator, exactly reproducible; fits the 2-read/1-write register-file port budget only if rd is read as a 3rd operand | a 3rd read port on the register file (or a separate accumulator register); the RTL must account for it |
| Result in rd | sext32 | later code can use rd as a normal XLEN value | none |
| Signedness | signed x signed | symmetric int8 weights and zero-point folded into the bias (section 2) so activations are used as signed | uint8 activations would need a different variant |
| Requant | one instruction with all parameters in rs2 (mult 32, shift 8, zp 8, min 8, max 8) | TFLite per-channel parameters fit exactly in 64 bits; clamp for fused ReLU/ReLU6 included | 64-bit multiply (acc x mult) + rounding in hardware: the longest path of the extension; est. 2+ cycles |
| Encoding | R-type in custom-0 (funct7 selects) | free opcode space, no conflict with RV64IMC or standard extensions | none |

### ldpi - post-increment load (X2, brief section 7)

```
ldpi rd, imm(rs1)        # custom-1 (0x2b), I-type, funct3 = 0
  a      = x[rs1]
  x[rd]  = mem64[a]                 # same alignment / access rules as `ld`
  x[rs1] = a + sext(imm12)          # only after the load succeeded
```

- **Motivation (measured, EXPERIMENTS.md section 4):** after X1 the 16-MAC inner loop is
  3 ld + 3 addi + 2 qmac8 + 1 bne; the three `addi` are pure pointer bumps.
- **Exceptions:** precise.  If the load traps (misaligned: cause 4, access fault:
  cause 5) neither rd nor rs1 is written; verified in `src/benchmarks/verify_ldpi.c`.
- **rd == rs1** is reserved and raises illegal-instruction (two writes to one register
  have no sensible result).  `rd = x0` is legal: it only bumps the pointer.
- **Why a separate opcode instead of a custom-0 funct7:** it is an I-type
  (12-bit immediate + base), the other two are R-type.
- **Hardware cost (to be measured at the RTL milestone):** a second register-file write
  port (rd and rs1 both written), an address-increment adder in the LSU / writeback of
  the load, and the corresponding hazard / forwarding logic: the rs1 update is available
  one stage earlier than the load data.  This is the dominant cost; the brief rates the
  feature as a medium-difficulty memory-system extension.
- **Scope:** only the 8-byte form with a free immediate is implemented (the kernels use
  +8); there is no byte/half/word variant and no pre-increment form.

## 4. Configurations (instruction-level comparison, brief section 4)

| Config | MAC loop | requant | loads / pointer bump | loop structure |
|---|---|---|---|---|
| B0 | scalar `mul`/`add` | software | ld + addi | naive, 1 output at a time |
| B1 | scalar `mul`/`add` | software | ld + addi | 2 output channels per pass, K unrolled x4 (best software-only) |
| X0 | **qmac8** | software | ld + addi | identical tile to B1 |
| X1 | **qmac8** | **qreq** | ld + addi | identical tile to B1 |
| X2 | qmac8 | qreq | **ldpi** | X1 loop, pointer bumps fused into the loads |
| X1T | qmac8 | qreq | ld + addi | X1 with cache-blocked loop order (software, not ISA) |
| X2T | qmac8 | qreq | **ldpi** | X1T + ldpi (the "final" configuration) |

B1, X0, X1, X2 share one loop skeleton, so `B1->X0` isolates qmac8, `X0->X1` isolates qreq
and `X1->X2` isolates ldpi (the brief's "incremental" evaluation).  X1T / X2T change only
the loop order so that the weights are reused from the D-cache.

## 5. Spike integration

`ext/qmac8.cc` is an `extension_t` built to `ext/libqmac8.so` and loaded with
`spike --extlib=ext/libqmac8.so --extension=qmac8`; no Spike source change.
Instructions are emitted from C with inline `.insn` (`src/nnaccel.h`), no function-call
overhead and no assembler change.  The qmac8 accumulator is carried in an `int64_t`
C variable (`qmac8_acc`) so the compiler does not insert sign-extensions after each
instruction (the instruction already writes a sign-extended 32-bit value).
`ext/cachesweep.cc` is a second extension (memory tracer) used for the cache study.

## 6. Verification strategy

| Level | Test | What it catches |
|---|---|---|
| instruction | `make spike_verify`: qmac8 directed + 2000 random, qreq 4824 golden vectors from an independent numpy model | semantic errors of the two instructions |
| instruction | `make spike_verify_ldpi`: directed, 4000 random, rd=x0, sequential stream, precise traps (misaligned, access fault), rd==rs1 | ldpi semantics incl. exceptions |
| kernel | `make spike_verify_gemm`: 120 random layer shapes (K = 1..300, extreme int8 values, random shifts/zero points/clamps) x 7 configs vs a plain-C reference | tail loops, K padding, blocking, ldpi loop |
| layer | `make layers`: 21 real layers x 5 configs vs TFLite | real data / quantization parameters |
| model | `make model_infer`: 37 ops chained x 7 configs, per-op checksum + logits of 10 images vs TFLite | chaining, pad/pool/add/im2col |
| simulator | `make test_cache` | the cache model used for the study |

