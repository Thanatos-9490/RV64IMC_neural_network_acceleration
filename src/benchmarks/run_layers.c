/* ---------------------------------------------------------------------------
 * run_layers.c - bit-exact check + benchmark of REAL ResNet-18 W8A8 layers.
 * ------------------------------------------------------------------------- */
#include <stdint.h>
#include "layer_kernels.h"
#include "layers.h"

#ifndef CFG
#define CFG CFG_B0
#endif

/* largest output: conv1 = 12544 x 64 = 802816 B */
#define OUT_MAX 802816
static int8_t out_buf[OUT_MAX] __attribute__((aligned(8)));

volatile int32_t sink;

int main(void)
{
    long bad = 0;
#ifdef ONLY_LAYER
    const int first = ONLY_LAYER, last = ONLY_LAYER + 1;
#else
    const int first = 0, last = NUM_LAYERS;
#endif
    for (int i = first; i < last; i++) {
        const layer_t *L = &LAYERS[i];
        const long n = (long)L->M * L->N;
        if (n > OUT_MAX) return 121;
        run_layer(L, CFG, out_buf);
        for (long j = 0; j < n; j++)
            bad += (out_buf[j] != L->y[j]);
    }
    sink = (int32_t)bad;
    return bad ? (int)(bad > 120 ? 120 : bad) : 0;
}
