/* ---------------------------------------------------------------------------
 * verify_qmac8.c - bare-metal functional verification of the qmac8 and qreq
 * ------------------------------------------------------------------------- */

#include <stdint.h>
#include "nnaccel.h"
#include "qreq_vectors.h"

/* Deterministic PRNG (reproducible across runs). */
static uint32_t rng_state = 0x5EED1234u;
static uint32_t rng_next(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static int8_t rng_i8(void) { return (int8_t)(rng_next() & 0xFF); }

/* ----------------------- scalar references ------------------------------- */

static int32_t dot_ref(const int8_t *a, const int8_t *w, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++)
        acc += (int32_t)a[i] * (int32_t)w[i];
    return acc;
}

/* Apply qmac8 to an 8-element array pair (with packing). */
static int32_t qmac8_arrays(int32_t acc, const int8_t a[8], const int8_t w[8])
{
    return qmac8(acc, nn_pack_i8x8(a), nn_pack_i8x8(w));
}

int main(void)
{
    int fails = 0;

    /* ----------------- directed qmac8 cases ----------------- */
    {
        int8_t a[8] = { 1, 2, 3, 4, -1, -2, -3, -4 };
        int8_t w[8] = { 10, 20, 30, 40, 5, 6, 7, 8 };
        int32_t got = qmac8_arrays(0, a, w);
        if (got != 230) fails++;
    }
    {
        int8_t a[8] = { 127, 127, 127, 127, 127, 127, 127, 127 };
        int8_t w[8] = { 127, 127, 127, 127, 127, 127, 127, 127 };
        if (qmac8_arrays(0, a, w) != 8 * 127 * 127) fails++;
    }
    {
        int8_t a[8] = { -128, -128, -128, -128, -128, -128, -128, -128 };
        int8_t w[8] = { 127, 127, 127, 127, 127, 127, 127, 127 };
        if (qmac8_arrays(0, a, w) != 8 * (-128) * 127) fails++;
    }
    {
        int8_t a[8] = { -128, 127, -1, 1, 0, 50, -50, 3 };
        int8_t w[8] = { 127, -128, -1, 1, 8, -2, 2, -3 };
        int32_t exp = dot_ref(a, w, 8);
        if (qmac8_arrays(0, a, w) != exp) fails++;
    }

    /* --------------- randomized qmac8 cases ----------------- */
    for (int iter = 0; iter < 2000; iter++) {
        int8_t a[8], w[8];
        int32_t acc = (int32_t)(rng_next() & 0xFFFFu) - 32768;
        for (int i = 0; i < 8; i++) { a[i] = rng_i8(); w[i] = rng_i8(); }
        int32_t exp = acc + dot_ref(a, w, 8);
        if (qmac8_arrays(acc, a, w) != exp) fails++;
    }

    /* ---- qreq: golden vectors (directed + random) from scripts/tflq.py,
     *      which is validated against the real TFLite interpreter ---- */
    for (int i = 0; i < NUM_QREQ_VECS; i++) {
        const qreq_vec_t *v = &QREQ_VECS[i];
        uint64_t p = nn_pack_req_params(v->mult, v->shift, v->zp, v->amin, v->amax);
        if ((int)qreq(v->acc, p) != (int)v->exp) fails++;
    }

    return fails ? 1 : 0;
}
