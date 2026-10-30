// ---------------------------------------------------------------------------
// qmac8.cc - RISC-V custom ISA extension for quantized-NN acceleration
//
// Adds two instructions to Spike, used by the RV64IMC quantized-NN term
// project (CV-Wally core).  The extension is implemented as a dynamically
// loadable Spike "extlib" (shared library), so it does not require
// recompiling Spike itself.
//
//   Instructions (R-type, opcode = custom-0 = 0x0b):
//
//     qmac8 rd, rs1, rs2      funct7 = 0x01, funct3 = 0x0
//     qreq  rd, rs1, rs2      funct7 = 0x02, funct3 = 0x0
//
//   X2 extension (I-type, opcode = custom-1 = 0x2b, funct3 = 0):
//
//     ldpi  rd, imm(rs1)      rd = mem64[rs1]; rs1 += imm   (post-increment load)
//
// Semantics
// ---------
// qmac8: 8 x INT8 packed multiply-accumulate.
//   rs1 = 8 packed INT8 activations (lane i at bits 8i..8i+7)
//   rs2 = 8 packed INT8 weights      (lane i at bits 8i..8i+7)
//   rd  = 32-bit accumulator (read-modify-write), sign-extended to XLEN
//   x[rd] = sext32( (int32)x[rd] + SUM_i sext(a[i]) * sext(w[i]) )
//
// qreq: TFLite-exact integer requantization of an INT32 accumulator to INT8.
//   rs1 = INT32 accumulator
//   rs2 = packed parameters:
//           bits 31..0  = multiplier  (INT32, Q31)
//           bits 39..32 = shift       (INT8; >0 left shift, <=0 right shift)
//           bits 47..40 = zero_point  (INT8)
//           bits 55..48 = act_min     (INT8, fused-activation lower clamp)
//           bits 63..56 = act_max     (INT8, fused-activation upper clamp)
//   rd  = sign-extended INT8 result
//   out = clamp( MultiplyByQuantizedMultiplier(acc, multiplier, shift)
//                + zero_point, act_min, act_max )
//
// The accumulator width (32 bits) matches the INT32 accumulator used by
// TFLite reference kernels for INT8 convolution / fully-connected layers.
// ---------------------------------------------------------------------------

#include "extension.h"
#include "mmu.h"
#include <cstdint>
#include "tfl_requant.h"

// Lane extraction helpers ---------------------------------------------------

static inline int8_t lane_i8(uint64_t packed, int lane)
{
  return (int8_t)((packed >> (8 * lane)) & 0xFF);
}

// Instruction handlers ------------------------------------------------------

// qmac8 rd, rs1, rs2
static reg_t qmac8_handler(processor_t* p, insn_t insn, reg_t pc)
{
  state_t* state = p->get_state();

  int32_t acc  = (int32_t)state->XPR[insn.rd()];
  uint64_t a   = state->XPR[insn.rs1()];
  uint64_t w   = state->XPR[insn.rs2()];

  for (int lane = 0; lane < 8; lane++) {
    int32_t ai = (int32_t)lane_i8(a, lane);
    int32_t wi = (int32_t)lane_i8(w, lane);
    acc += ai * wi;
  }

  reg_t result = (reg_t)(int64_t)acc;        // sign-extend 32 -> 64
  state->log_reg_write[insn.rd() << 4] = {result, 0};
  state->XPR.write(insn.rd(), result);

  return pc + 4;
}

// qreq rd, rs1, rs2
static reg_t qreq_handler(processor_t* p, insn_t insn, reg_t pc)
{
  state_t* state = p->get_state();

  int32_t  acc    = (int32_t)state->XPR[insn.rs1()];
  uint64_t params = state->XPR[insn.rs2()];

  int32_t mult  = (int32_t)(params & 0xFFFFFFFFull);
  int     shift = (int8_t)((params >> 32) & 0xFF);
  int     zp    = (int8_t)((params >> 40) & 0xFF);
  int     amin  = (int8_t)((params >> 48) & 0xFF);
  int     amax  = (int8_t)((params >> 56) & 0xFF);

  int8_t out = tfl_requant_i8(acc, mult, shift, zp, amin, amax);

  reg_t result = (reg_t)(int64_t)out;         // sign-extend INT8 -> 64
  state->log_reg_write[insn.rd() << 4] = {result, 0};
  state->XPR.write(insn.rd(), result);

  return pc + 4;
}

// ldpi rd, imm(rs1)   (X2: load 64 bit with post-increment of the base) ------
//
//   opcode custom-1 = 0x2b, I-type, funct3 = 0
//   addr  = x[rs1]
//   x[rd] = mem64[addr]            (8-byte load, same alignment/trap rules as ld)
//   x[rs1] = addr + sext(imm12)    (after the load succeeded)
//
// Precise exceptions: the load is performed first; if it traps, NEITHER register
// is modified.  rd == rs1 is reserved (the two writes would collide) and raises
// an illegal-instruction exception; rd == x0 is allowed (discard data, bump
// pointer).  Two destination registers => the core needs a second write port.
static reg_t ldpi_handler(processor_t* p, insn_t insn, reg_t pc)
{
  state_t* state = p->get_state();
  int rd = insn.rd(), rs1 = insn.rs1();
  if (rd == rs1)
    throw trap_illegal_instruction(insn.bits());

  reg_t addr = state->XPR[rs1];
  uint64_t val = p->get_mmu()->load<uint64_t>(addr);   // may trap: no state change yet
  reg_t base = addr + (reg_t)(int64_t)insn.i_imm();

  state->log_reg_write[rd << 4] = {val, 0};
  state->XPR.write(rd, val);
  state->log_reg_write[rs1 << 4] = {base, 0};
  state->XPR.write(rs1, base);
  return pc + 4;
}

// Disassembly ---------------------------------------------------------------

// Self-contained register name table (avoids linking against libdisasm).
static const char* xreg_name[32] = {
  "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
  "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
  "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
  "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
};

struct xrd_arg  : public arg_t { std::string to_string(insn_t i) const override { return xreg_name[i.rd()]; } };
struct xrs1_arg : public arg_t { std::string to_string(insn_t i) const override { return xreg_name[i.rs1()]; } };
struct xrs2_arg : public arg_t { std::string to_string(insn_t i) const override { return xreg_name[i.rs2()]; } };
struct imm_rs1_arg : public arg_t {
  std::string to_string(insn_t i) const override {
    return std::to_string((int)i.i_imm()) + "(" + xreg_name[i.rs1()] + ")";
  }
};

// Extension class -----------------------------------------------------------

class qmac8_ext_t : public extension_t
{
 public:
  const char* name() const override { return "qmac8"; }

  std::vector<insn_desc_t> get_instructions(const processor_t&) override
  {
    // R-type: [funct7][rs2][rs1][funct3][rd][opcode]
    // mask keeps funct7 (31:25), funct3 (14:12), opcode (6:0).
    const insn_bits_t mask = 0xFE00707Full;

    std::vector<insn_desc_t> insns = {
      // qmac8 : match = funct7=0x01, funct3=0x0, opcode=0x0b
      { 0x0200000Bull, mask,
        qmac8_handler, qmac8_handler, qmac8_handler, qmac8_handler,
        qmac8_handler, qmac8_handler, qmac8_handler, qmac8_handler },
      // qreq  : match = funct7=0x02, funct3=0x0, opcode=0x0b
      { 0x0400000Bull, mask,
        qreq_handler, qreq_handler, qreq_handler, qreq_handler,
        qreq_handler, qreq_handler, qreq_handler, qreq_handler },
      // ldpi  : I-type, opcode custom-1 = 0x2b, funct3 = 0 (imm is free)
      { 0x0000002Bull, 0x0000707Full,
        ldpi_handler, ldpi_handler, ldpi_handler, ldpi_handler,
        ldpi_handler, ldpi_handler, ldpi_handler, ldpi_handler },
    };
    return insns;
  }

  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override
  {
    static xrd_arg  rd;
    static xrs1_arg rs1;
    static xrs2_arg rs2;
    static imm_rs1_arg mem;

    std::vector<disasm_insn_t*> insns = {
      new disasm_insn_t("ldpi",  0x0000002Bull, 0x0000707Full, {&rd, &mem}),
      new disasm_insn_t("qmac8", 0x0200000Bull, 0xFE00707Full, {&rd, &rs1, &rs2}),
      new disasm_insn_t("qreq",  0x0400000Bull, 0xFE00707Full, {&rd, &rs1, &rs2}),
    };
    return insns;
  }
};

REGISTER_EXTENSION(qmac8, []() { static qmac8_ext_t ext; return &ext; })
