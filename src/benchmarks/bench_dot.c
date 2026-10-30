/* ---------------------------------------------------------------------------
 * bench_dot.c - dot-product microbenchmark (baseline vs qmac8).
 * ------------------------------------------------------------------------- */

#include <stdint.h>
#include "nnaccel.h"
#include "kernels.h"

#ifndef USE_QMAC8
#define USE_QMAC8 0
#endif

#define LEN     64
#define REPEAT  1000

volatile int32_t sink;

static void measure_begin(void) { __asm__ volatile(".globl b_begin\nb_begin:\n" ::: "memory"); }
static void measure_end(void)   { __asm__ volatile(".globl b_end\nb_end:\n"     ::: "memory"); }

int main(void)
{
    static int8_t a[LEN] __attribute__((aligned(8)));
    static int8_t w[LEN] __attribute__((aligned(8)));

    for (int i = 0; i < LEN; i++) {
        a[i] = (int8_t)((i * 7 + 3) & 0xFF);
        w[i] = (int8_t)((i * 13 + 5) & 0xFF);
    }

    int32_t result = 0;

    measure_begin();
    for (int r = 0; r < REPEAT; r++) {
#if USE_QMAC8
        result = dot_product_i8_qmac8(a, w, LEN);
#else
        result = dot_product_i8_ref(a, w, LEN);
#endif
    }
    measure_end();

    sink = result;

    int32_t expected = dot_product_i8_ref(a, w, LEN);
    return (result == expected) ? 0 : 1;
}
