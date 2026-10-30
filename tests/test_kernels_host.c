/* ---------------------------------------------------------------------------
 * test_kernels_host.c - host-side functional verification of the accelerated
 * ------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdint.h>
#include "nnaccel.h"
#include "kernels.h"
#include "qreq_vectors.h"

/* Deterministic PRNG (xorshift-ish LCG) for reproducible randomized tests. */
static uint32_t rng_state = 0x12345678u;
static uint32_t rng_next(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static int8_t rng_i8(void) { return (int8_t)(rng_next() & 0xFF); }

static int failures = 0;

#define CHECK(cond, msg) do {                                  \
    if (!(cond)) {                                             \
        printf("FAIL: %s (line %d)\n", msg, __LINE__);         \
        failures++;                                            \
    }                                                          \
} while (0)

/* ------------------------------ dot product ----------------------------- */

static void test_dot_product(void)
{
    /* Directed: mixed signs, no accumulator wrap. */
    {
        int8_t a[8] = {1, 2, 3, 4, -1, -2, -3, -4};
        int8_t w[8] = {10, 20, 30, 40, 5, 6, 7, 8};
        int32_t ref = dot_product_i8_ref(a, w, 8);
        int32_t got = dot_product_i8_qmac8(a, w, 8);
        printf("dot directed ref=%d got=%d\n", ref, got);
        CHECK(ref == got, "dot_product directed");
    }

    /* Randomized across many lengths (tests tail handling + packing). */
    for (int len = 1; len <= 64; len++) {
        int8_t a[64] __attribute__((aligned(8)));
        int8_t w[64] __attribute__((aligned(8)));
        for (int i = 0; i < len; i++) { a[i] = rng_i8(); w[i] = rng_i8(); }
        int32_t ref = dot_product_i8_ref(a, w, len);
        int32_t got = dot_product_i8_qmac8(a, w, len);
        if (ref != got) { printf("FAIL dot len=%d ref=%d got=%d\n", len, ref, got); failures++; }
    }
    printf("dot_product randomized done\n");
}

/* -------------------------------- requant -------------------------------- */

static void test_requant(void)
{
    int bad = 0;
    for (int i = 0; i < NUM_QREQ_VECS; i++) {
        const qreq_vec_t *v = &QREQ_VECS[i];
        uint64_t p = nn_pack_req_params(v->mult, v->shift, v->zp, v->amin, v->amax);
        if (qreq(v->acc, p) != v->exp) { if (bad < 5) printf("FAIL qreq vec %d\n", i); bad++; }
    }
    printf("requant: %d golden vectors, %d mismatches\n", NUM_QREQ_VECS, bad);
    failures += bad;
}

/* ----------------------------------- FC ---------------------------------- */

static void test_fc(void)
{
    enum { OC = 4, IC = 33 };
    int8_t  input[IC] __attribute__((aligned(8)));
    int8_t  weights[OC][IC] __attribute__((aligned(8)));
    int32_t bias[OC];
    int8_t  out_ref[OC], out_q[OC];

    for (int i = 0; i < IC; i++) input[i] = rng_i8();
    for (int o = 0; o < OC; o++) {
        bias[o] = (int32_t)(rng_next() & 0x3FF) - 512;
        for (int i = 0; i < IC; i++) weights[o][i] = rng_i8();
    }

    int32_t multiplier = 12345, shift = 16; int8_t zp = -8;

    fully_connected_i8_ref(input, &weights[0][0], bias, out_ref, OC, IC, multiplier, shift, zp);
    fully_connected_i8_qmac8(input, &weights[0][0], bias, out_q, OC, IC, multiplier, shift, zp);

    for (int o = 0; o < OC; o++) {
        printf("FC[%d] ref=%d q=%d\n", o, out_ref[o], out_q[o]);
        CHECK(out_ref[o] == out_q[o], "FC mismatch");
    }
    printf("fc done\n");
}

/* ---------------------------------- GEMM --------------------------------- */

static void test_gemm(void)
{
    enum { M = 8, N = 8, K = 17 };
    int8_t  A[M][K] __attribute__((aligned(8)));
    int8_t  B[K][N] __attribute__((aligned(8)));
    int8_t  Bp[N][K] __attribute__((aligned(8)));
    int32_t C_ref[M][N], C_q[M][N];

    for (int i = 0; i < M; i++)
        for (int l = 0; l < K; l++) A[i][l] = rng_i8();
    for (int l = 0; l < K; l++)
        for (int j = 0; j < N; j++) B[l][j] = rng_i8();

    gemm_i8_ref(&A[0][0], &B[0][0], &C_ref[0][0], M, N, K);
    gemm_pack_b(&B[0][0], &Bp[0][0], N, K);
    gemm_i8_qmac8(&A[0][0], &Bp[0][0], &C_q[0][0], M, N, K);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            if (C_ref[i][j] != C_q[i][j]) {
                printf("FAIL GEMM [%d][%d] ref=%d got=%d\n", i, j, C_ref[i][j], C_q[i][j]);
                failures++;
            }
    printf("gemm done\n");
}

int main(void)
{
    test_dot_product();
    test_requant();
    test_fc();
    test_gemm();

    if (failures == 0) { printf("ALL HOST KERNEL TESTS PASSED\n"); return 0; }
    printf("HOST KERNEL TESTS FAILED: %d\n", failures);
    return 1;
}
