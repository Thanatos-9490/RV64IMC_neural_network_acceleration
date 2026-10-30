#ifndef NNACCEL_H
#define NNACCEL_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Portable packing helpers.
 * ---------------------------------------------------------------------- */

/* Pack 8 signed INT8 values into one 64-bit word (little-endian lanes):
 *   bits  7:0   = x[0]
 *   bits 15:8   = x[1]
 *   ...
 *   bits 63:56  = x[7]
 */
static inline uint64_t nn_pack_i8x8(const int8_t x[8])
{
    uint64_t p = 0;
    for (int i = 0; i < 8; i++)
        p |= ((uint64_t)(uint8_t)x[i]) << (8 * i);
    return p;
}

/* Pack the requantization parameters consumed by qreq (TFLite semantics):
 *   bits 31..0  = multiplier  (INT32, Q31)
 *   bits 39..32 = shift       (INT8, >0 left shift, <=0 right shift)
 *   bits 47..40 = zero_point  (INT8)
 *   bits 55..48 = act_min     (INT8, fused-activation lower clamp)
 *   bits 63..56 = act_max     (INT8, fused-activation upper clamp)
 */
static inline uint64_t nn_pack_req_params(int32_t multiplier, int shift,
                                          int zero_point, int act_min,
                                          int act_max)
{
    return ((uint64_t)(uint32_t)multiplier)
         | ((uint64_t)(uint8_t)shift << 32)
         | ((uint64_t)(uint8_t)zero_point << 40)
         | ((uint64_t)(uint8_t)act_min << 48)
         | ((uint64_t)(uint8_t)act_max << 56);
}

/* -------------------------------------------------------------------------
 * Hardware intrinsics.
 *
 * On RV64IMC these are implemented by src/nnaccel.S, which emits the real
 * custom instructions:
 *
 *     qmac8 rd, rs1, rs2   : 8 x INT8 packed multiply-accumulate
 *     qreq  rd, rs1, rs2   : INT32 -> INT8 requantization
 *
 * On the host (x86), a software model is linked instead so the same kernel
 * source can be verified natively.
 * ---------------------------------------------------------------------- */

#ifdef __riscv
/* Real custom instructions, emitted inline (no call overhead), R-type, opcode
 * custom-0 (0x0b):  qmac8 = funct7 0x01,  qreq = funct7 0x02.
 * qmac8 reads AND writes rd (accumulator), hence the "+r" constraint. */
static inline __attribute__((always_inline))
int32_t qmac8(int32_t acc, uint64_t packed_a, uint64_t packed_w)
{
    __asm__ (".insn r 0x0b, 0x0, 0x01, %0, %1, %2"
             : "+r"(acc) : "r"(packed_a), "r"(packed_w));
    return acc;
}

static inline __attribute__((always_inline))
int8_t qreq(int32_t acc, uint64_t packed_params)
{
    int64_t out;
    __asm__ (".insn r 0x0b, 0x0, 0x02, %0, %1, %2"
             : "=r"(out) : "r"((int64_t)acc), "r"(packed_params));
    return (int8_t)out;
}

/* 64-bit-register variants used by the layer kernels.
 *
 * qmac8 reads only rd[31:0] and writes sext32(result) to rd, so a value that
 * lives in a 64-bit register is ALWAYS a valid sign-extended int32.  Declaring
 * the accumulator int32_t (above) makes GCC re-sign-extend (c.addiw rd,0) after
 * every qmac8 because it cannot know that; keeping it int64_t in the loop
 * removes those two instructions from the 16-MAC inner loop (11 -> 9).
 * Semantics are unchanged, only the C type of the carrier differs.          */
static inline __attribute__((always_inline))
int64_t qmac8_acc(int64_t acc, uint64_t packed_a, uint64_t packed_w)
{
    __asm__ (".insn r 0x0b, 0x0, 0x01, %0, %1, %2"
             : "+r"(acc) : "r"(packed_a), "r"(packed_w));
    return acc;
}

static inline __attribute__((always_inline))
int8_t qreq_acc(int64_t acc, uint64_t packed_params)
{
    int64_t out;
    __asm__ (".insn r 0x0b, 0x0, 0x02, %0, %1, %2"
             : "=r"(out) : "r"(acc), "r"(packed_params));
    return (int8_t)out;
}

/* X2: 64-bit load with post-increment.  v = *(uint64_t*)p; p += 8.
 * `=&r` keeps rd != rs1 (the instruction traps on rd == rs1).  The "m" input
 * tells GCC the asm reads *p, so it is not reordered across stores to it. */
static inline __attribute__((always_inline))
uint64_t ldpi8(const int8_t **p)
{
    const int8_t *q = *p;
    uint64_t v;
    __asm__ (".insn i 0x2b, 0x0, %0, 8(%1)"
             : "=&r"(v), "+r"(q) : "m"(*(const uint64_t (*)[1])q));
    *p = q;
    return v;
}
#else
static inline uint64_t ldpi8(const int8_t **p)
{ uint64_t v = *(const uint64_t *)*p; *p += 8; return v; }
/* host build: bit-exact software model in src/qmac8_model.c */
int32_t qmac8(int32_t acc, uint64_t packed_a, uint64_t packed_w);
int8_t qreq(int32_t acc, uint64_t packed_params);
static inline int64_t qmac8_acc(int64_t acc, uint64_t a, uint64_t w)
{ return qmac8((int32_t)acc, a, w); }
static inline int8_t qreq_acc(int64_t acc, uint64_t p)
{ return qreq((int32_t)acc, p); }
#endif

#endif /* NNACCEL_H */
