/* ---------------------------------------------------------------------------
 * model_ops.h - the non-GEMM operators of ResNet-18 W8A8 (int8, NHWC), written
 * ------------------------------------------------------------------------- */
#ifndef MODEL_OPS_H
#define MODEL_OPS_H
#include <stdint.h>

/* in [C][H][W] -> out [H][W][C] */
void op_transpose(const int8_t *in, int8_t *out, int C, int H, int W);

/* in [ih][iw][c] -> out [oh][ow][c], border filled with zp (quantized PAD) */
void op_pad(const int8_t *in, int8_t *out, int ih, int iw, int c,
            int oh, int ow, int pad_t, int pad_l, int zp);

/* im2col: col[M = oh*ow][Kp], tap order (ky,kx,c), border = in_zp (SAME),
 * K..Kp-1 = 0.  K = kh*kw*ic. */
void op_im2col(const int8_t *in, int8_t *col, int ih, int iw, int ic, int kh,
               int kw, int stride, int pad_t, int pad_l, int oh, int ow,
               int in_zp, int Kp);

void op_maxpool(const int8_t *in, int8_t *out, int ih, int iw, int c, int k,
                int stride, int oh, int ow, int act_min, int act_max);

/* TFLite int8 ADD (left_shift 20, inputs rescaled to 2*max(scale)) */
void op_add(const int8_t *in1, const int8_t *in2, int8_t *out, int n,
            int off1, int off2, int lshift, int m1, int s1, int m2, int s2,
            int mo, int so, int out_zp, int act_min, int act_max);

/* global average pool over ih*iw (VALID, single window) */
void op_avgpool(const int8_t *in, int8_t *out, int ih, int iw, int c,
                int act_min, int act_max);

#endif
