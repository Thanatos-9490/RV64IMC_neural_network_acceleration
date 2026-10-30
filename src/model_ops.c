/* No libc on target: stop GCC from turning the copy/fill loops into memcpy /
 * memset calls (they would be undefined symbols). */
#pragma GCC optimize("no-tree-loop-distribute-patterns")

#include "model_ops.h"
#include "tfl_requant.h"

#define NOINLINE __attribute__((noinline))

/* byte / 64-bit-word copy and fill.  Built with -fno-tree-loop-distribute-patterns
 * so GCC does not turn these loops into memcpy/memset calls (no libc here). */
static inline void copy_bytes(int8_t *d, const int8_t *s, long n)
{
    if (((((uintptr_t)d) | ((uintptr_t)s) | (uintptr_t)n) & 7) == 0) {
        uint64_t *dw = (uint64_t *)d;
        const uint64_t *sw = (const uint64_t *)s;
        for (long i = 0; i < n / 8; i++)
            dw[i] = sw[i];
    } else {
        for (long i = 0; i < n; i++)
            d[i] = s[i];
    }
}

static inline void fill_bytes(int8_t *d, int v, long n)
{
    if (((((uintptr_t)d) | (uintptr_t)n) & 7) == 0) {
        uint64_t x = (uint8_t)v;
        x |= x << 8;
        x |= x << 16;
        x |= x << 32;
        uint64_t *dw = (uint64_t *)d;
        for (long i = 0; i < n / 8; i++)
            dw[i] = x;
    } else {
        for (long i = 0; i < n; i++)
            d[i] = (int8_t)v;
    }
}

NOINLINE void op_transpose(const int8_t *in, int8_t *out, int C, int H, int W)
{
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++) {
            const int8_t *s = in + ((long)c * H + y) * W;
            int8_t *d = out + (long)y * W * C + c;
            for (int x = 0; x < W; x++, d += C)
                *d = s[x];
        }
}

NOINLINE void op_pad(const int8_t *in, int8_t *out, int ih, int iw, int c,
                     int oh, int ow, int pad_t, int pad_l, int zp)
{
    const long orow = (long)ow * c, irow = (long)iw * c;
    const long left = (long)pad_l * c, right = orow - left - irow;
    for (int y = 0; y < oh; y++) {
        int8_t *d = out + y * orow;
        int sy = y - pad_t;
        if (sy < 0 || sy >= ih) {
            fill_bytes(d, zp, orow);
        } else {
            fill_bytes(d, zp, left);
            copy_bytes(d + left, in + sy * irow, irow);
            fill_bytes(d + left + irow, zp, right);
        }
    }
}

NOINLINE void op_im2col(const int8_t *in, int8_t *col, int ih, int iw, int ic,
                        int kh, int kw, int stride, int pad_t, int pad_l,
                        int oh, int ow, int in_zp, int Kp)
{
    const long K = (long)kh * kw * ic, rowbytes = (long)kw * ic;
    for (int oy = 0; oy < oh; oy++) {
        for (int ox = 0; ox < ow; ox++) {
            int8_t *dst = col + ((long)oy * ow + ox) * Kp;
            const int x0 = ox * stride - pad_l;
            const int x_in = (x0 >= 0 && x0 + kw <= iw);     /* whole row inside */
            for (int ky = 0; ky < kh; ky++) {
                const int iy = oy * stride - pad_t + ky;
                if (iy < 0 || iy >= ih) {
                    fill_bytes(dst, in_zp, rowbytes);
                } else if (x_in) {
                    copy_bytes(dst, in + ((long)iy * iw + x0) * ic, rowbytes);
                } else {
                    for (int kx = 0; kx < kw; kx++) {
                        const int ix = x0 + kx;
                        if (ix < 0 || ix >= iw)
                            fill_bytes(dst + (long)kx * ic, in_zp, ic);
                        else
                            copy_bytes(dst + (long)kx * ic,
                                       in + ((long)iy * iw + ix) * ic, ic);
                    }
                }
                dst += rowbytes;
            }
            for (long k = K; k < Kp; k++)                    /* zero K padding */
                *dst++ = 0;
        }
    }
}

NOINLINE void op_maxpool(const int8_t *in, int8_t *out, int ih, int iw, int c,
                         int k, int stride, int oh, int ow, int act_min,
                         int act_max)
{
    (void)ih;
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++) {
            int8_t *d = out + ((long)oy * ow + ox) * c;
            for (int ch = 0; ch < c; ch++) {
                int m = -128;
                for (int ky = 0; ky < k; ky++) {
                    const int8_t *s = in + (((long)oy * stride + ky) * iw
                                            + (long)ox * stride) * c + ch;
                    for (int kx = 0; kx < k; kx++, s += c)
                        if (*s > m) m = *s;
                }
                if (m < act_min) m = act_min;
                if (m > act_max) m = act_max;
                d[ch] = (int8_t)m;
            }
        }
}

NOINLINE void op_add(const int8_t *in1, const int8_t *in2, int8_t *out, int n,
                     int off1, int off2, int lshift, int m1, int s1, int m2,
                     int s2, int mo, int so, int out_zp, int act_min, int act_max)
{
    for (int i = 0; i < n; i++) {
        const int32_t x1 = (in1[i] + off1) * (1 << lshift);
        const int32_t x2 = (in2[i] + off2) * (1 << lshift);
        const int32_t a = tfl_mulq(x1, m1, s1);              /* shifts are <= 0 */
        const int32_t b = tfl_mulq(x2, m2, s2);
        int32_t v = tfl_mulq(a + b, mo, so) + out_zp;
        if (v < act_min) v = act_min;
        if (v > act_max) v = act_max;
        out[i] = (int8_t)v;
    }
}

NOINLINE void op_avgpool(const int8_t *in, int8_t *out, int ih, int iw, int c,
                         int act_min, int act_max)
{
    const int cnt = ih * iw;
    for (int ch = 0; ch < c; ch++) {
        int32_t acc = 0;
        for (int p = 0; p < cnt; p++)
            acc += in[(long)p * c + ch];
        acc = acc > 0 ? (acc + cnt / 2) / cnt : (acc - cnt / 2) / cnt;
        if (acc < act_min) acc = act_min;
        if (acc > act_max) acc = act_max;
        out[ch] = (int8_t)acc;
    }
}
