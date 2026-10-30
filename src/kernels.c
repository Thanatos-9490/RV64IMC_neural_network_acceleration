/* ---------------------------------------------------------------------------
 * kernels.c - reference and qmac8/qreq-accelerated kernels for the
 * ------------------------------------------------------------------------- */

#include "nnaccel.h"
#include "tfl_requant.h"

/* ------------------------- dot product (inner MAC) ----------------------- */

int32_t dot_product_i8_ref(const int8_t *a, const int8_t *w, int32_t len)
{
    int32_t acc = 0;
    for (int32_t i = 0; i < len; i++)
        acc += (int32_t)a[i] * (int32_t)w[i];
    return acc;
}

int32_t dot_product_i8_qmac8(const int8_t *a, const int8_t *w, int32_t len)
{
    int32_t acc = 0;
    int32_t i = 0;

    for (; i + 8 <= len; i += 8) {
        uint64_t pa = *(const uint64_t *)(a + i);  /* one 8-byte load */
        uint64_t pw = *(const uint64_t *)(w + i);
        acc = qmac8(acc, pa, pw);
    }
    for (; i < len; i++)
        acc += (int32_t)a[i] * (int32_t)w[i];

    return acc;
}

/* --------------------------- requantization ------------------------------ */

int8_t requantize_i8_ref(int32_t acc, int32_t multiplier, int shift,
                         int8_t zero_point)
{
    /* TFLite-exact, int8 range clamp (see tfl_requant.h) */
    return tfl_requant_i8(acc, multiplier, shift, zero_point, -128, 127);
}

/* ------------------------- fully connected layer ------------------------- */

void fully_connected_i8_ref(const int8_t *input,
                            const int8_t *weights,      /* [OC][IC] */
                            const int32_t *bias,
                            int8_t *output,
                            int32_t oc, int32_t ic,
                            int32_t multiplier, int shift, int8_t zero_point)
{
    for (int32_t o = 0; o < oc; o++) {
        const int8_t *wrow = &weights[o * ic];
        int32_t acc = bias[o];
        for (int32_t i = 0; i < ic; i++)
            acc += (int32_t)input[i] * (int32_t)wrow[i];
        output[o] = requantize_i8_ref(acc, multiplier, shift, zero_point);
    }
}

void fully_connected_i8_qmac8(const int8_t *input,
                              const int8_t *weights,    /* [OC][IC] */
                              const int32_t *bias,
                              int8_t *output,
                              int32_t oc, int32_t ic,
                              int32_t multiplier, int shift, int8_t zero_point)
{
    uint64_t params = nn_pack_req_params(multiplier, shift, zero_point, -128, 127);

    for (int32_t o = 0; o < oc; o++) {
        const int8_t *wrow = &weights[o * ic];
        int32_t acc = bias[o];
        int32_t i = 0;

        for (; i + 8 <= ic; i += 8) {
            uint64_t pa = *(const uint64_t *)(input + i);
            uint64_t pw = *(const uint64_t *)(wrow + i);
            acc = qmac8(acc, pa, pw);
        }
        for (; i < ic; i++)
            acc += (int32_t)input[i] * (int32_t)wrow[i];

        output[o] = qreq(acc, params);
    }
}

/* B1: software-optimized RV64IMC (no custom instructions).
 * 4-way unrolled inner loop with a walking pointer, so loop/branch and
 * address overhead is amortised.  Same semantics as fully_connected_i8_ref. */
void fully_connected_i8_opt(const int8_t *input,
                            const int8_t *weights,      /* [OC][IC] */
                            const int32_t *bias,
                            int8_t *output,
                            int32_t oc, int32_t ic,
                            int32_t multiplier, int shift, int8_t zero_point)
{
    for (int32_t o = 0; o < oc; o++) {
        const int8_t *w = &weights[o * ic];
        const int8_t *a = input;
        const int8_t *end4 = a + (ic & ~3);
        const int8_t *end  = a + ic;
        int32_t acc = bias[o];

        while (a < end4) {
            acc += (int32_t)a[0] * w[0] + (int32_t)a[1] * w[1]
                 + (int32_t)a[2] * w[2] + (int32_t)a[3] * w[3];
            a += 4; w += 4;
        }
        while (a < end)
            acc += (int32_t)*a++ * (int32_t)*w++;

        output[o] = requantize_i8_ref(acc, multiplier, shift, zero_point);
    }
}

/* X0: qmac8 only; requantization stays in software (incremental step 1). */
void fully_connected_i8_qmac8_swreq(const int8_t *input,
                                    const int8_t *weights,  /* [OC][IC] */
                                    const int32_t *bias,
                                    int8_t *output,
                                    int32_t oc, int32_t ic,
                                    int32_t multiplier, int shift,
                                    int8_t zero_point)
{
    for (int32_t o = 0; o < oc; o++) {
        const int8_t *wrow = &weights[o * ic];
        int32_t acc = bias[o];
        int32_t i = 0;

        for (; i + 8 <= ic; i += 8) {
            uint64_t pa = *(const uint64_t *)(input + i);
            uint64_t pw = *(const uint64_t *)(wrow + i);
            acc = qmac8(acc, pa, pw);
        }
        for (; i < ic; i++)
            acc += (int32_t)input[i] * (int32_t)wrow[i];

        output[o] = requantize_i8_ref(acc, multiplier, shift, zero_point);
    }
}

/* --------------------------------- GEMM ---------------------------------- */

void gemm_i8_ref(const int8_t *A,          /* [M][K] */
                 const int8_t *B,          /* [K][N] */
                 int32_t *C,               /* [M][N] */
                 int32_t m, int32_t n, int32_t k)
{
    for (int32_t i = 0; i < m; i++)
        for (int32_t j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int32_t l = 0; l < k; l++)
                acc += (int32_t)A[i * k + l] * (int32_t)B[l * n + j];
            C[i * n + j] = acc;
        }
}

/* Pack B from [K][N] to Bp [N][K] (transpose) so the reduction dim is
 * contiguous.  Performed once offline, just like TFLite weight packing. */
void gemm_pack_b(const int8_t *B, int8_t *Bp, int32_t n, int32_t k)
{
    for (int32_t j = 0; j < n; j++)
        for (int32_t l = 0; l < k; l++)
            Bp[j * k + l] = B[l * n + j];
}

void gemm_i8_qmac8(const int8_t *A,        /* [M][K] */
                   const int8_t *Bp,       /* [N][K] packed */
                   int32_t *C,             /* [M][N] */
                   int32_t m, int32_t n, int32_t k)
{
    for (int32_t i = 0; i < m; i++)
        for (int32_t j = 0; j < n; j++) {
            const int8_t *arow = &A[i * k];
            const int8_t *brow = &Bp[j * k];
            int32_t acc = 0;
            int32_t l = 0;

            for (; l + 8 <= k; l += 8) {
                uint64_t pa = *(const uint64_t *)(arow + l);
                uint64_t pb = *(const uint64_t *)(brow + l);
                acc = qmac8(acc, pa, pb);
            }
            for (; l < k; l++)
                acc += (int32_t)arow[l] * (int32_t)brow[l];

            C[i * n + j] = acc;
        }
}
