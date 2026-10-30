# EXPERIMENTS - what was measured, how, and what it does and does not show

Everything is reproducible: `make` (verification, no TensorFlow) and
`make profile && python3 scripts/summarize.py` (profiling); the full list of commands is in section 8.  All tables below are
copied from `results/processed/summary.md`, which is generated.

## 1. Functional correctness (brief section 5, "Functional")

| Check | Command | Result |
|---|---|---|
| `qmac8`: 4 directed + 2000 random cases vs scalar reference, on Spike | `make spike_verify` | pass |
| `qreq`: 4824 golden vectors (directed edges + random) from the independent numpy model `scripts/tflq.py`, on Spike | `make spike_verify` | pass, 0 mismatches |
| Software model of both instructions vs reference (host) | `make host_test` | pass |
| **All 21 ResNet-18 layers, 4 configs, vs the TFLite interpreter, byte-for-byte** | `make layers` | exit 0 in B0, B1, X0, X1 |
| **Whole network, 37 ops chained**: per-op checksum (image 0) + 1000 logits x 10 images, B0/B1/X0/X1/X1T/X2/X2T | `make model_infer` | exit 0 in all 7 configs (B0 B1 X0 X1 X1T X2 X2T), 0 mismatching ops, 0 mismatching logits, top-1 9/10 = TFLite 9/10 |
| `ldpi` (X2): directed, 4000 random, rd=x0, stream, precise traps (misaligned, access fault), rd==rs1 | `make spike_verify_ldpi` | pass; 3 mutants of the Spike handler all caught |
| **Randomized GEMM**: 120 random shapes (K=1..300, extreme int8, random shifts/zp/clamps) x 7 configs vs plain-C reference | `make spike_verify_gemm` | 0 mismatches; 3 mutants of the kernels all caught |
| Second instruction counter: hardware `instret` vs PC histogram, whole model, 7 configs | `python3 scripts/instret_check.py` | identical after removing the 3 fixed read-pair instructions per op (residual 0) |
| Cache simulator: 5 hand-computed probes (LRU/FIFO/PLRU) + equality with `spike --dc` on a real layer, 4 geometries | `make test_cache` | all pass |
| No FP instruction in any built ELF | objdump scan of model_x1, layers_x1, layers_b0 | 0 FP instructions |

Negative control (the check can fail): changing one rounding constant in
`src/tfl_requant.h` makes the layer run fail with 120+ mismatching bytes
(exit 120); it was restored afterwards.

What `make layers` proves: for one real image, each layer's int8 output equals the
TFLite output (each layer's input taken from the TFLite run).  `make model_infer`
proves the stronger statement: the ops run **chained** (each op consumes the previous
op's own output, from a static arena), so errors would accumulate; the result is
still byte-identical.  Negative controls on the new non-GEMM kernels (each mutation
is detected, `make model_infer` fails with exit 120/75): AVG_POOL negative-rounding
branch, ADD output rounding, MAX_POOL initial value, im2col border value, PAD
value.  One mutation was NOT detected: the AVG_POOL *positive* rounding branch
(`acc + cnt/2` -> `+1`), because for the 49-element window of this one image no sum
falls where the two differ.  That branch is therefore verified by reading the TFLite
source (copied verbatim) but not by this test.

Images: 10 (one per Imagenette class) for the logits; the per-op checksum uses image 0
only.  JPEG decode / resize / normalisation are done offline and are not part of the
measured network.  The model input is the int8 NCHW tensor of the TFLite graph.

## 2. Workload (brief section 2B)

ResNet-18, 224x224: 20 CONV_2D + 1 FC = **1814.1 MMAC, 100% of all MACs**
(`results/processed/resnet18_analysis.txt`).  Conv layers are lowered to GEMM by
im2col; the 21 GEMMs span K=64..4608 and M=1..12544.

## 3. Instruction-level results (brief sections 3-4)

Measured with Spike's PC histogram (`-g`), classified from the disassembly.  The
method was validated against the full `-l` trace on a layer (identical totals).
Two views, both generated: `results/processed/summary.md` (21 stand-alone GEMM
layers, `make profile`) and `results/processed/model_summary.md` (the chained
37-op model, `make model_profile`).  The GEMM part of the chained run equals the
sum of the stand-alone layers (diff <= 21 instructions out of 10^9..10^10).

**Change since the first milestone.**  The qmac8 accumulator is now carried in an
`int64_t` C variable (`qmac8_acc` in `src/nnaccel.h`) instead of `int32_t`.  The
instruction is unchanged (it reads rd[31:0] and writes sext32), but GCC no longer
inserts two `c.addiw` sign-extensions per iteration: the X1 inner loop is
**9 instructions for 16 MACs** (3 ld, 3 addi, 2 qmac8, 1 bne) instead of 11.  This
is the variant the first report flagged as "not adopted".  `-DACC32` rebuilds the old
loop and reproduces the old count exactly (op05/X1: 82 316 907 vs 82 316 906 before),
so the earlier numbers (5.78x) remain reproducible; the new ones are below.

### Whole model (37 ops), one image

| Config | Description | Dyn. insns | non-GEMM share | MAC/insn | vs B0 | vs B1 |
|---|---|---:|---:|---:|---:|---:|
| B0 | baseline RV64IMC | 12 971 051 944 | 0.9% | 0.140 | 1.00x | 0.58x |
| B1 | software-optimised | 7 540 125 048 | 1.5% | 0.241 | 1.72x | 1.00x |
| X0 | B1 loop + qmac8 | 1 289 121 952 | 8.6% | 1.407 | 10.06x | 5.85x |
| X1 | X0 + qreq | 1 169 118 495 | 9.5% | 1.552 | 11.09x | **6.45x** |
| X1T | X1, cache-blocked loop order | 1 159 348 802 | 9.6% | 1.565 | 11.19x | 6.50x |
| X2 | X1 + `ldpi` post-increment load | 825 802 649 | 13.5% | 2.197 | 15.71x | **9.13x** |
| X2T | X1T + `ldpi` | 818 638 576 | 13.6% | 2.216 | 15.84x | 9.21x |

- Incremental: **qmac8** (B1->X0) = 5.85x, **qreq** (X0->X1) = 1.10x, **ldpi** (X1->X2) = 1.42x (whole model; 1.48x on the GEMM part alone).
- The 111.4 M instructions of the 6 non-GEMM operator kinds are identical in every
  config: add 65.8 M, im2col 23.1 M, maxpool 19.9 M, pad 1.7 M, transpose 0.76 M,
  avgpool 0.13 M.  After X1 they are 9.5% of the model: **Amdahl limit** -- GEMM-only
  B1->X1 is 7.02x, whole model 6.45x.  Next software targets would be ADD (5.6% of
  X1: three `tfl_mulq` per element) and im2col (2.0%).
- GEMM-only (21 layers): B0 12.86 G, B1 7.43 G, X0 1.178 G, X1 1.058 G (B1->X1 7.02x;
  per layer 6.44x (op14) to 7.10x; >=2x met on every layer, vs B0 10.2x-12.4x).
- X1T retires 0.8% fewer instructions than X1 only because it hoists the per-channel
  qreq operand/bias out of the m loop; its purpose is the data cache (section 6).

### Instruction mix (whole model)

| Config | load | store | ALU | multiply | branch+jump | qmac8 | qreq |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 28.3% | 0.1% | 43.4% | 14.0% | 14.3% | - | - |
| B1 | 36.7% | 0.2% | 35.5% | 24.1% | 3.5% | - | - |
| X0 | 29.2% | 0.6% | 40.5% | 0.4% | 11.7% | 17.6% | - |
| X1 | 31.1% | 0.6% | 37.0% | 0.2% | 11.4% | 19.4% | 0.2% |

Custom-instruction share in X1: 19.4% qmac8 + 0.2% qreq; in X2: qmac8 28-32%, ldpi 47.7% (of the GEMM code), qreq 0.35%.

### Requantization (brief section 3: "instructions/output element")

qreq removes **about 48 instructions per output element** in every layer
(47.7-49.0).  That is constant, so its relative value falls as K grows: 49% of X0 for
the K=64 layer (op14), 32.7% for conv1, but only 1.8% for the K=4608 layers.

### Code size (kernel `.text`, one configuration compiled alone, bytes)

B0 370, B1 814, X0 608, X1 234, X2 206 (X0 was 612 with the int32 carrier).  X1 is the
smallest because the 8-wide MAC removes the unrolling and tail code.

### Microbenchmark (brief: "one microbenchmark")

64-element dot product x 1000: B0 462 001 insns -> X1 81 001 (**5.70x**), MAC/insn 0.139 -> 0.790
(`make micro`; unchanged by this milestone, it uses `src/kernels.c`).

## 4. Bottlenecks (brief section 2B: at least three)

Measured on the whole model (numbers from section 3 and `model_summary.md`):

1. **Scalar multiply/MAC loop (B0).**  14.0% multiplies, 2.0 loads per MAC,
   MAC/insn 0.14.  Removed by qmac8 (multiply share -> 0.2%).
2. **Load/store pressure (B0, B1).**  ld+st per MAC is 2.0 (B0) and 1.5 (B1).
   qmac8 reduces it to 0.20 because one 64-bit load feeds 8 MACs.
3. **Loop / index overhead (B0).**  Branch+jump 14.3% of B0, 3.5% after tiling (B1);
   11.4% in X1 because the useful work shrank.
4. **Requantization (small after X0).**  ~48 insns/output, 1.10x overall.
5. **Non-GEMM operators (new, after X1).**  Untouched by the ISA extension: 111 M
   instructions = 9.5% of X1 (ADD 5.6%, im2col 2.0%, maxpool 1.7%).
6. **Data-cache misses (new, after X1; section 6).**  Same bytes touched, 6.5x fewer
   instructions, so MPKI rises from 3.7 (B1) to 23.9 (X1) at 16 KB.

**What limited X1, and what limits X2.**  The X1 inner loop (9 instructions for 16
MACs) was 3 ld + 3 addi (pointer bumps) + 2 qmac8 + 1 bne: only 2 of 9 retired
instructions do MACs.  X2 replaces each ld+addi pair by one `ldpi` and the loop becomes

```
 ldpi t,(A)+8 ; ldpi u,(W0)+8 ; ldpi v,(W1)+8 ; qmac8 acc0 ; qmac8 acc1 ; bne
```

**6 instructions for 16 MACs** (verified in `objdump` of `gemm_x2`): 1.42x fewer
instructions on the whole model (1.48x on the GEMM part), MAC/insn 1.55 -> 2.20.
The remaining mix in X2 (GEMM code): ldpi 47.7%, qmac8 31.8%, branch 16.1%, ALU 2.3%.
What is left to remove, in order: (1) the loop branch (16%: a hardware loop would take
it away, brief section 7), (2) the 3 loads per 2 qmac8: a fused load+MAC or a wider
register tile (4 channels per pass would load A once for 4 qmac8 instead of 2), (3) the
11.4% of the model that is not GEMM (section 3).  These are candidates, **not evaluated
here**.  Whether memory latency/bandwidth limits the loop is a cycle question: section 6
measures the miss side, section 7 turns it into a stated model.

## 5. Limitations (read before citing any number)

- **Instruction counts, not cycles.**  No CPI, cycle, Fmax or area exist yet.
  Speedups in sections 3-4 are dynamic-instruction ratios.  A `qmac8` that takes more
  cycles than an `add` would reduce the real speedup.
- **Ten images** for the logits check, **one image** for per-op checksums and for all
  instruction/cache counts (the network has no data-dependent control flow except
  max-pool/clamp, so counts vary by only a few instructions between images, but this
  was only measured on the one).
- **Pre-processing is offline** (decode, resize, normalise); the int8 input tensor is
  given.  The AVG_POOL positive-rounding branch is not exercised by the test image.
- **Static memory**: 1.85 MB activation arena + 1.82 MB im2col buffer, plus 11.7 MB of
  weights in `.rodata`; the im2col buffer is a real cost of the lowering that the
  stand-alone layer profile did not include (23 M instructions to fill, and it is
  also cache traffic).
- **B1 is one reasonable software optimisation, not the optimum.**  A better
  hand-tuned B1 would shrink the headline 6.45x.
- **Accuracy** was measured on 500 Imagenette images (10 classes, 1000-way argmax),
  sampling error about +-4 pt; the W8A8 vs FP32 difference (+1.4 pt) is within noise.

## 6. Cache design exploration (brief section 7: "Cache Design Exploration")

Generated tables: `results/processed/cache_summary.md`; figures:
`results/figures/cache_geometry.png`, `cache_policy.png`.  Reproduce: `make test_cache`,
`make cache`.

**Method.**  `ext/cachesweep.cc` is a Spike extension that hooks the memory tracer
and feeds every data load/store of the chained inference to 31 independent
set-associative, write-back, write-allocate cache models in one run, with
LRU / PLRU (binary tree, as CV-Wally `cacheLRU.sv`) / FIFO / RANDOM replacement.
Accesses are attributed to graph ops through 1-byte stores to `phase_mark`;
the verification code is excluded.  Cold cache, one image, **data accesses only**
(instruction fetch not modelled).  CV-Wally's default D-cache is 4 way x 4 KB,
64 B lines, PLRU (`config/rv64gc`); the "16K-4w-64B-plru" row is that cache.

**Simulator validation** (`tests/test_cachesweep.py`, run by `make test_cache`):
5 hand-computed access patterns for LRU/FIFO/PLRU pass, and a `srand` policy that
copies Spike's own `cache_sim_t` reproduces `spike --dc` exactly (reads, writes, read
misses, write misses, writebacks) for 4 geometries on a real layer.  The PLRU and
FIFO/LRU models are validated only by the directed probes, not against RTL.

**Cycle model (assumption, not a measurement).**  cycles = instructions + misses x
(LAT + line/8), LAT = 10 or 40, blocking cache, CPI 1 on hits, write-backs hidden.
This is not CV-Wally's CPI.  Treat the "model CPI" column as an illustration of
sensitivity; the miss rates and traffic are the measured part.

### Finding 1: sensitivity depends on the workload (B1 vs X1 vs X1T)

| 16 KB, 4-way, 64 B, PLRU | B1 | X1 | X1T |
|---|---:|---:|---:|
| D-cache miss rate | 1.000% | 7.533% | 3.157% |
| misses per 1000 instructions | 3.7 | 23.9 | 10.0 |
| memory traffic (fills + write-backs) | 1797 MB | 1810 MB | 768 MB |
| model CPI, LAT 10 / LAT 40 | 1.066 / 1.177 | 1.430 / 2.148 | 1.179 / 1.478 |
| speedup vs B1 incl. misses, LAT 10 / LAT 40 | 1x | 4.81x / 3.53x | 5.88x / 5.18x |
| instruction-count speedup vs B1 (no misses) | 1x | 6.45x | 6.50x |

B1 barely notices its cache (1.0% miss at 16 KB, 1.6% even at 2 KB).  After qmac8 the program
touches the same bytes in 6.5x fewer instructions, so the miss *rate per access*
jumps 7.5x and the cache becomes first-order: under the model, the ISA gain 6.45x
shrinks to 3.5-4.8x.  This is the "new bottleneck" the brief asks to identify.

**Why.**  In `gemm_x1` (and B1) the whole weight matrix (N*Kp bytes: 36 KB for
op05-op09, 144 KB, 576 KB ... 2.3 MB for op29-op33) is streamed once per output row
`m`.  For layers where it exceeds the cache that is a cyclic scan, which LRU-like
policies turn into ~100% misses on the weight stream (per-layer table: 8.1-9.4% miss
rate in every large layer, 0.2-0.4% in the two layers whose weights fit: op02, op14).
X1T processes a block of 8 KB of A rows against each channel pair so each weight is
fetched once per block: miss rate falls to 0.8-1.5% on op05-op20, 4.0% on
op21-op28 (op02 gets slightly worse, 0.35% -> 0.49%), and **does not change on op29/op32/op33** (Kp = 4608, block = 1 row).
X1T is bit-exact (`make model_infer`).

### Finding 2: geometry (X1 | X1T, PLRU, miss rate)

| axis | values | X1 | X1T |
|---|---|---|---|
| size (4-way, 64 B) | 2 / 4 / 8 / 16 / 32 / 64 KB | 11.4 / 10.0 / 8.9 / 7.5 / 6.7 / 5.2 % | 10.4 / 8.4 / 6.7 / 3.2 / 2.7 / 2.5 % |
| associativity (16 KB) | 1 / 2 / 4 / 8 / 16 way | 9.8 / 7.7 / 7.5 / 7.3 / 7.3 % | 5.2 / 3.2 / 3.2 / 3.1 / 3.0 % |
| line (16 KB, 4-way) | 16 / 32 / 64 / 128 B | 30.0 / 15.0 / 7.5 / 3.8 % | 12.2 / 6.2 / 3.2 / 1.7 % |

- X1T has a sharp knee between 8 and 16 KB (6.7% -> 3.2%): the 8 KB block of A plus the
  weight pair fits in 16 KB but not in 8 KB.  X1 improves slowly and steadily with size
  because its working set (hundreds of KB) never fits.  CV-Wally's 16 KB is just past
  the knee for the blocked kernel.
- Associativity: 1 -> 2 way matters (direct-mapped conflicts between A and W: +29%
  misses for X1, +61% for X1T); beyond 2 ways the gain is at most 5% (X1) / 7% (X1T).
- Line size: the miss rate halves each time the line doubles (streaming access, one
  miss per line), while total traffic stays ~1.8 GB (X1) because every byte is used;
  for X1T traffic grows slightly with the line (733 -> 831 MB).  Whether a 128 B line
  is a win depends on the bus width / fill latency, i.e. on the RTL milestone; the
  model above charges line/8 cycles per fill.

### Finding 3: replacement policy (miss count relative to PLRU, 4-way, 64 B)

| size | X1: LRU / FIFO / RAND | X1T: LRU / FIFO / RAND |
|---|---|---|
| 4 KB | 1.00 / 1.08 / 1.11 | 1.00 / 1.12 / 1.15 |
| 8 KB | 1.00 / 1.09 / 1.07 | 1.04 / 1.12 / 1.03 |
| 16 KB | 1.00 / 1.10 / 1.09 | 0.96 / 1.21 / 1.30 |
| 32 KB | 1.01 / 1.08 / 0.97 | 1.01 / 1.17 / 1.20 |

The best-to-worst spread across the four policies at the same geometry is 9-14% for
B1 and X1 and **15-44% for X1T** (35% at 16 KB 4-way, 44% at 16 KB 8-way): the policy
matters most for the program that was tuned to fit the cache, because its reuse
distance sits close to the capacity, whereas for the streaming programs every policy
loses the same lines.  PLRU stays within 2% of true LRU for B1 and X1 (LRU has 0-1.8% more misses).  For X1T
the difference is larger and has both signs (true LRU: -7% misses at 16 KB 8-way, +4%
at 8 KB 4-way), so CV-Wally's cheap tree-PLRU is adequate but is not always the best.
FIFO and RANDOM have more misses than PLRU in 34 of the 36 (geometry, workload) cells;
the two exceptions are RANDOM at 32 KB for X1 (-3%) and B1 (-1%), so the ranking is not
uniform.  Differences of ~1% are within what this single-image, cold-start
simulation can resolve and are not claimed.

### What this does and does not show

- Shows (measured): miss rates, miss counts and memory traffic per cache geometry and
  policy for the real chained workload; that the ISA extension turns the workload from
  cache-insensitive to cache-bound; that loop order recovers a large part of it.
- Does not show: cycles (the penalty model is assumed), instruction-cache effects,
  write-buffer/bus behaviour, or CV-Wally RTL behaviour.  The RTL milestone should
  re-measure at least the 16 KB / PLRU point to calibrate the model.
- X1T is a software change, so "cache-policy-only" is not the whole story: the larger
  gain is from matching the loop order to the cache, which the sweep itself exposed.

## 7. ISA improvement vs microarchitectural improvement (brief section 4)

Generated: `results/processed/perf_model.md` (`python3 scripts/perf_model.py`).  The
brief requires the two to be distinguished; this repository separates them as follows.

- **ISA improvement = measured**: dynamic instructions per class (section 3).  This is
  what the extension removes, independent of any pipeline.
- **Microarchitectural effect = model**: cycles = sum(count x cost) + branch penalty +
  load-use + misses x (LAT + 8).  Costs: ALU/ld/st/branch 1, taken-branch +2 (90% of
  branches), load-use +1 on half of the loads, qmac8 `Q` cycles (scanned 1..4), qreq 2
  (or 4), miss LAT = 10 (or 40) + 8.  Inputs are measured, the costs are **assumptions**.

| Config | insns (whole model) | ISA-only speedup vs B1 | modelled cycle speedup vs B1 (qmac8 1 cy, LAT 10) | cycle / ISA |
|---|---:|---:|---:|---:|
| B1 | 7.540 G | 1.00x | 1.00x | 1.00 |
| X0 | 1.289 G | 5.85x | 4.40x | 0.75 |
| X1 | 1.169 G | 6.45x | 4.72x | 0.73 |
| X1T | 1.159 G | 6.50x | 5.53x | 0.85 |
| X2 | 0.826 G | 9.13x | 5.67x | 0.62 |
| X2T | 0.819 G | 9.21x | 6.84x | 0.74 |

Reading: the instruction-count gain overstates what a pipeline would deliver: the model
retains 62-85% of it.  The largest losses are the cache misses of X1/X2 (7.5% miss
rate; recovered by the blocking in X1T/X2T) and the taken-branch cost, which grows in
relative terms as the loop body shrinks (a 6-instruction loop pays a 2-cycle bubble per
iteration: that is the argument for a hardware loop).  The B1 -> X2T gain under the
stated assumptions ranges from 4.3x (qmac8 = 4 cycles, LAT 40) to 7.6x (qmac8 = 1 cycle,
no misses); `perf_model.md` section 3 has the full table.  **This range is the
uncertainty that the RTL milestone has to remove**: it needs the real qmac8 / qreq
latency and a measured CPI.

Per output element (GEMM part): 5175 instructions (B0), 2990 (B1), 474 (X0), 426 (X1),
288 (X2), 285 (X2T); modelled cycles per output (whole model) 7301, 3984, 906, 844, 703,
582.

Cache behaviour of the new kernels: `ldpi` does not change the address stream, so the miss
counts of X2/X2T equal X1/X1T up to < 1% (16 KB: 7.47% and 3.17%).

## 8. Reproduction (all commands from the repository root)

```bash
make                           # extlib, host test, qmac8/qreq/ldpi/GEMM verification,
                               #   21 layers x 5 configs, full model x 7 configs
make test_cache                # cache simulator tests
make profile                   # per-layer profile (5 configs)   ~6 min
make model_profile             # whole-model profile (7 configs) ~8 min
python3 scripts/instret_check.py   # instret cross-check       ~3 min
make cache                     # cache sweep B1/X1/X1T (+ scripts/cache_sweep.py --cfgs x2,x2t)
python3 scripts/summarize.py; python3 scripts/summarize_model.py
python3 scripts/perf_model.py; python3 scripts/cache_summary.py; python3 scripts/make_figures.py
```

Re-export of the model (TensorFlow, `conda activate aca_term_prj`):
`python scripts/export_layers.py && python scripts/export_model.py`.
Environment: Spike 1.1.1-dev (`/opt/riscv`), riscv64-unknown-elf-gcc, Python 3 with
numpy/matplotlib; TensorFlow 2.15 only for export/quantization.
