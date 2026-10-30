/* ---------------------------------------------------------------------------
 * tfl_requant.h - TFLite/gemmlowp integer requantization, bit-exact.
 * ------------------------------------------------------------------------- */
#ifndef TFL_REQUANT_H
#define TFL_REQUANT_H

#include <stdint.h>

/* round-to-nearest, ties away from zero, of (a*b) / 2^31 (saturating). */
static inline int32_t tfl_srdhm(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN)
        return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int64_t nudge = (ab >= 0) ? ((int64_t)1 << 30) : (1 - ((int64_t)1 << 30));
    int64_t t = ab + nudge;
    /* C division truncates toward zero, as in the reference */
    return (int32_t)(t / ((int64_t)1 << 31));
}

/* x / 2^e rounded to nearest, ties away from zero.  0 <= e <= 31. */
static inline int32_t tfl_rdbp(int32_t x, int e)
{
    int32_t mask = (int32_t)(((int64_t)1 << e) - 1);
    int32_t rem = x & mask;
    int32_t thr = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> e) + (rem > thr ? 1 : 0);
}

static inline int32_t tfl_mulq(int32_t x, int32_t m, int shift)
{
    int ls = shift > 0 ? shift : 0;
    int rs = shift > 0 ? 0 : -shift;
    if (rs > 31) rs = 31;
    return tfl_rdbp(tfl_srdhm((int32_t)((uint32_t)x << ls), m), rs);
}

static inline int8_t tfl_requant_i8(int32_t acc, int32_t mult, int shift,
                                    int zp, int act_min, int act_max)
{
    int32_t v = tfl_mulq(acc, mult, shift) + zp;
    if (v < act_min) v = act_min;
    if (v > act_max) v = act_max;
    return (int8_t)v;
}

#endif /* TFL_REQUANT_H */
