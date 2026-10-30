#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>

/* dot product */
int32_t dot_product_i8_ref(const int8_t *a, const int8_t *w, int32_t len);
int32_t dot_product_i8_qmac8(const int8_t *a, const int8_t *w, int32_t len);

/* requantization */
int8_t requantize_i8_ref(int32_t acc, int32_t multiplier, int shift,
                         int8_t zero_point);

/* fully connected */
void fully_connected_i8_ref(const int8_t *input, const int8_t *weights,
                            const int32_t *bias, int8_t *output,
                            int32_t oc, int32_t ic,
                            int32_t multiplier, int shift, int8_t zero_point);

void fully_connected_i8_qmac8(const int8_t *input, const int8_t *weights,
                              const int32_t *bias, int8_t *output,
                              int32_t oc, int32_t ic,
                              int32_t multiplier, int shift, int8_t zero_point);

/* B1: software-optimized (unrolled, no custom insns) */
void fully_connected_i8_opt(const int8_t *input, const int8_t *weights,
                            const int32_t *bias, int8_t *output,
                            int32_t oc, int32_t ic,
                            int32_t multiplier, int shift, int8_t zero_point);

/* X0: qmac8 only, software requantization (incremental step 1) */
void fully_connected_i8_qmac8_swreq(const int8_t *input, const int8_t *weights,
                                    const int32_t *bias, int8_t *output,
                                    int32_t oc, int32_t ic,
                                    int32_t multiplier, int shift,
                                    int8_t zero_point);

/* GEMM */
void gemm_i8_ref(const int8_t *A, const int8_t *B, int32_t *C,
                 int32_t m, int32_t n, int32_t k);
void gemm_pack_b(const int8_t *B, int8_t *Bp, int32_t n, int32_t k);
void gemm_i8_qmac8(const int8_t *A, const int8_t *Bp, int32_t *C,
                   int32_t m, int32_t n, int32_t k);

#endif /* KERNELS_H */
