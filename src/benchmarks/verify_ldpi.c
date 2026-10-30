/* ---------------------------------------------------------------------------
 * verify_ldpi.c - directed + randomized verification of `ldpi rd, imm(rs1)`
 * (X2: 64-bit load, then rs1 += imm).  Runs on Spike; exit 0 = pass.
 * ------------------------------------------------------------------------- */
#include <stdint.h>

#define LDPI(rd, base, imm)                                                    \
    __asm__ volatile (".insn i 0x2b, 0x0, %0, " #imm "(%1)"                    \
                      : "=&r"(rd), "+r"(base) : : "memory")

static uint64_t buf[64] __attribute__((aligned(64)));
static uint32_t rs = 0xC0FFEE11u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs; }

/* RW variant: rd is also an input, so after a trap we can see it was NOT written */
#define LDPI_RW(rd, base, imm)                                                 \
    __asm__ volatile (".insn i 0x2b, 0x0, %0, " #imm "(%1)"                    \
                      : "+r"(rd), "+r"(base) : : "memory")

/* trapbuf: [0]=saved t1 [1]=saved t0 [2]=last mcause [3]=trap count.
 * The handler preserves every register (mscratch holds &trapbuf), skips the
 * 4-byte faulting instruction and returns. */
volatile uint64_t trapbuf[4];
#define trap_cause trapbuf[2]
#define trap_count trapbuf[3]

void trap_handler(void);
__asm__ (".align 2\n.globl trap_handler\ntrap_handler:\n"
         "  csrrw t0, mscratch, t0\n"        /* t0 = &trapbuf, mscratch = old t0 */
         "  sd    t1, 0(t0)\n"
         "  csrr  t1, mscratch\n  sd t1, 8(t0)\n"
         "  csrr  t1, mcause\n    sd t1, 16(t0)\n"
         "  ld    t1, 24(t0)\n    addi t1, t1, 1\n  sd t1, 24(t0)\n"
         "  csrr  t1, mepc\n      addi t1, t1, 4\n  csrw mepc, t1\n"
         "  csrw  mscratch, t0\n"            /* ready for the next trap */
         "  ld    t1, 0(t0)\n     ld t0, 8(t0)\n"
         "  mret\n");

int main(void)
{
    int fails = 0;
    for (int i = 0; i < 64; i++)
        buf[i] = 0x0101010101010101ull * (uint64_t)(i + 1) ^ ((uint64_t)i << 40);

    /* directed: +8 (the kernel use), 0, positive, negative, 12-bit limits */
    {
        const uint64_t *p = &buf[4]; uint64_t v; uintptr_t b = (uintptr_t)p;
        LDPI(v, b, 8);     if (v != buf[4] || b != (uintptr_t)&buf[5]) fails++;
        LDPI(v, b, 0);     if (v != buf[5] || b != (uintptr_t)&buf[5]) fails++;
        LDPI(v, b, 16);    if (v != buf[5] || b != (uintptr_t)&buf[7]) fails++;
        LDPI(v, b, -24);   if (v != buf[7] || b != (uintptr_t)&buf[4]) fails++;
        b = (uintptr_t)&buf[40];
        LDPI(v, b, 2040);  if (v != buf[40] || b != (uintptr_t)&buf[40] + 2040) fails++;
        b = (uintptr_t)&buf[63];
        LDPI(v, b, -2048); if (v != buf[63] || b != (uintptr_t)&buf[63] - 2048) fails++;
    }
    /* rd = x0: no data, base still incremented */
    {
        uintptr_t b = (uintptr_t)&buf[10];
        __asm__ volatile (".insn i 0x2b, 0x0, x0, 8(%0)" : "+r"(b) : : "memory");
        if (b != (uintptr_t)&buf[11]) fails++;
    }
    /* randomized: random aligned address and a random multiple-of-8 immediate */
    for (int it = 0; it < 4000; it++) {
        int idx = rnd() % 64;
        uintptr_t b = (uintptr_t)&buf[idx], b0 = b; uint64_t v;
        switch (rnd() % 4) {
        case 0: LDPI(v, b, 8);   b0 += 8;   break;
        case 1: LDPI(v, b, 64);  b0 += 64;  break;
        case 2: LDPI(v, b, -8);  b0 -= 8;   break;
        default: LDPI(v, b, 0);             break;
        }
        if (v != buf[idx] || b != b0) fails++;
    }
    /* sequential stream, as the GEMM kernel uses it */
    {
        const int8_t *p = (const int8_t *)buf; uint64_t sum = 0, ref = 0;
        for (int i = 0; i < 64; i++) { uint64_t v; uintptr_t b = (uintptr_t)p;
            LDPI(v, b, 8); p = (const int8_t *)b; sum += v; ref += buf[i]; }
        if (sum != ref || p != (const int8_t *)(buf + 64)) fails++;
    }

    /* traps: install handler (direct mode), then provoke */
    __asm__ volatile ("csrw mtvec, %0" : : "r"((uintptr_t)trap_handler));
    __asm__ volatile ("csrw mscratch, %0" : : "r"((uintptr_t)trapbuf));
    {   /* misaligned address: Spike (no misaligned support) raises cause 4 */
        uintptr_t b = (uintptr_t)&buf[3] + 1, b0 = b; uint64_t v = 0x1234;
        trap_count = 0;
        LDPI_RW(v, b, 8);
        if (trap_count != 1 || trap_cause != 4) fails++;        /* load addr misaligned */
        if (b != b0 || v != 0x1234) fails++;                    /* precise: nothing written */
    }
    {   /* access fault: address 0 is unmapped -> cause 5 */
        uintptr_t b = 0, b0 = b; uint64_t v = 0x5678;
        trap_count = 0;
        LDPI_RW(v, b, 8);
        if (trap_count != 1 || trap_cause != 5) fails++;
        if (b != b0 || v != 0x5678) fails++;
    }
    {   /* rd == rs1 is illegal (cause 2).  Encode by hand: ldpi a0, 8(a0). */
        trap_count = 0;
        register uintptr_t a0 __asm__("a0") = (uintptr_t)&buf[1];
        __asm__ volatile (".insn i 0x2b, 0x0, a0, 8(a0)" : "+r"(a0) : : "memory");
        if (trap_count != 1 || trap_cause != 2) fails++;
        if (a0 != (uintptr_t)&buf[1]) fails++;
    }
    return fails ? (fails > 100 ? 100 : fails) : 0;
}
