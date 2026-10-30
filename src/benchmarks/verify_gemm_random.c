/* ---------------------------------------------------------------------------
 * verify_gemm_random.c - randomized GEMM-level verification on Spike.
 * ------------------------------------------------------------------------- */
#include <stdint.h>
#include "layer_kernels.h"
#include "tfl_requant.h"
#include "nnaccel.h"

#define MAXM 12
#define MAXN 12
#define MAXKP 304

static int8_t A[MAXM * MAXKP] __attribute__((aligned(8)));
static int8_t W[MAXN * MAXKP] __attribute__((aligned(8)));
static int32_t BIAS[MAXN], MULT[MAXN];
static int8_t SHIFT[MAXN];
static uint64_t PACK[MAXN];
static int8_t OUT[MAXM * MAXN] __attribute__((aligned(8)));
static int8_t REF[MAXM * MAXN];

static uint32_t rs = 0xBADC0DE5u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs; }
static int8_t rnd_i8(int extreme)
{
    uint32_t r = rnd();
    if (extreme && (r & 7) == 0) return (r & 8) ? -128 : 127;
    return (int8_t)(r >> 8);
}

int main(void)
{
    int fails = 0;
    static const int cfgs[] = {CFG_B0, CFG_B1, CFG_X0, CFG_X1, CFG_X1T, CFG_X2, CFG_X2T};
    for (int it = 0; it < 120; it++) {
        const int M = 1 + rnd() % MAXM;
        const int N = 2 * (1 + rnd() % (MAXN / 2));
        const int K = 1 + rnd() % 300;
        const int Kp = (K + 7) / 8 * 8;
        const int extreme = it & 1;
        for (int m = 0; m < M; m++)
            for (int k = 0; k < Kp; k++)
                A[m * Kp + k] = k < K ? rnd_i8(extreme) : 0;
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < Kp; k++)
                W[n * Kp + k] = k < K ? rnd_i8(extreme) : 0;
            BIAS[n] = (int32_t)(rnd() % 200001) - 100000;
            MULT[n] = (int32_t)(0x40000000u + (rnd() & 0x3FFFFFFFu));   /* [0.5,1) Q31 */
            SHIFT[n] = (int8_t)((int)(rnd() % 14) - 11);               /* -11 .. +2 */
        }
        const int zp = (int)(rnd() % 256) - 128;
        int amin = -128, amax = 127;
        if (amin < -128) amin = -128;
        switch (rnd() % 3) { case 1: amin = zp; break; case 2: amin = zp; amax = zp + 6; break; }
        if (amax > 127) amax = 127;
        if (amin > amax) amin = amax;
        for (int n = 0; n < N; n++)
            PACK[n] = nn_pack_req_params(MULT[n], SHIFT[n], zp, amin, amax);

        int ok_range = 1;
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int64_t acc = BIAS[n];
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[m * Kp + k] * (int32_t)W[n * Kp + k];
                if (acc > INT32_MAX || acc < INT32_MIN) ok_range = 0;
                REF[m * N + n] = tfl_requant_i8((int32_t)acc, MULT[n], SHIFT[n], zp, amin, amax);
            }
        if (!ok_range) { fails++; continue; }

        layer_t L = {"rnd", M, N, K, Kp, zp, amin, amax, A, W, 0, BIAS, MULT, SHIFT, PACK};
        for (unsigned c = 0; c < sizeof cfgs / sizeof cfgs[0]; c++) {
            for (int i = 0; i < M * N; i++) OUT[i] = 0x55;
            run_layer(&L, cfgs[c], OUT);
            for (int i = 0; i < M * N; i++)
                if (OUT[i] != REF[i]) { fails++; break; }
        }
    }
    return fails > 100 ? 100 : fails;
}
