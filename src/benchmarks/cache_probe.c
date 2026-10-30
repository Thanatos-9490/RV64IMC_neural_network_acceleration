/* ---------------------------------------------------------------------------
 * cache_probe.c - directed access pattern with a hand-computed answer for the
 * cache simulator (ext/cachesweep.cc).  tests/test_cachesweep.py runs it.
 * ------------------------------------------------------------------------- */
#include <stdint.h>

volatile uint8_t phase_mark[64] __attribute__((aligned(64)));
static volatile uint8_t probe[8 * 64] __attribute__((aligned(64)));

#ifndef BLOCK
#define BLOCK 0
#endif
#define LD(i) ((void)*(volatile uint8_t *)(base + (i) * 64))

int main(void)
{
    const uintptr_t base = (uintptr_t)probe;
    phase_mark[0] = 1;
#if BLOCK == 0
    LD(0); LD(1); LD(2); LD(3); LD(0); LD(4); LD(1);      /* A B C D A E B */
#else
    LD(0); LD(1); LD(0); LD(2); LD(0); LD(1);             /* A B A C A B */
#endif
    phase_mark[63] = 1;
    return 0;
}
