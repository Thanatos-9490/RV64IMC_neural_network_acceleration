/* ---------------------------------------------------------------------------
 * layer_t.h - descriptor of one GEMM-form layer (conv via im2col, or FC).
 * ------------------------------------------------------------------------- */
#ifndef LAYER_T_H
#define LAYER_T_H
#include <stdint.h>

typedef struct {
    const char *name;
    int M, N, K, Kp;   /* out[M][N] = A[M][Kp] * W[N][Kp]^T; K true, Kp = K rounded up to 8 */
    int out_zp, act_min, act_max;
    const int8_t *a, *w, *y;
    const int32_t *bias, *mult;
    const int8_t *shift;
    const uint64_t *packed;          /* per-channel qreq operand */
} layer_t;

#endif
