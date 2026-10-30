#include "layer_kernels.h"
#include "nnaccel.h"
#include "tfl_requant.h"

#define LD64(p) (*(const uint64_t *)(p))

static inline int8_t req_sw(const layer_t *L, int n, int32_t acc)
{
    return tfl_requant_i8(acc, L->mult[n], L->shift[n], L->out_zp,
                          L->act_min, L->act_max);
}

/* ---- B0: naive ----------------------------------------------------------- */
static void gemm_b0(const layer_t *L, int8_t *out)
{
    const int M = L->M, N = L->N, K = L->K, Kp = L->Kp;
    for (int m = 0; m < M; m++) {
        const int8_t *a = L->a + (long)m * Kp;
        for (int n = 0; n < N; n++) {
            const int8_t *w = L->w + (long)n * Kp;
            int32_t acc = L->bias[n];
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[k] * (int32_t)w[k];
            out[(long)m * N + n] = req_sw(L, n, acc);
        }
    }
}

/* ---- B1: 2-channel tile, 4x unrolled, software requant -------------------- */
static void gemm_b1(const layer_t *L, int8_t *out)
{
    const int M = L->M, N = L->N, K = L->K, Kp = L->Kp;
    const int K4 = K & ~3;
    for (int m = 0; m < M; m++) {
        const int8_t *a = L->a + (long)m * Kp;
        for (int n = 0; n < N; n += 2) {
            const int8_t *w0 = L->w + (long)n * Kp;
            const int8_t *w1 = w0 + Kp;
            int32_t s0 = L->bias[n], s1 = L->bias[n + 1];
            int k = 0;
            for (; k < K4; k += 4) {
                int32_t a0 = a[k], a1 = a[k + 1], a2 = a[k + 2], a3 = a[k + 3];
                s0 += a0 * w0[k] + a1 * w0[k + 1] + a2 * w0[k + 2] + a3 * w0[k + 3];
                s1 += a0 * w1[k] + a1 * w1[k + 1] + a2 * w1[k + 2] + a3 * w1[k + 3];
            }
            for (; k < K; k++) {
                s0 += (int32_t)a[k] * w0[k];
                s1 += (int32_t)a[k] * w1[k];
            }
            out[(long)m * N + n]     = req_sw(L, n, s0);
            out[(long)m * N + n + 1] = req_sw(L, n + 1, s1);
        }
    }
}

/* ---- X0 / X1: same tile, qmac8 over the zero-padded Kp (no tail loop) -----
 * The accumulators are int64_t carriers of an int32 value (see nnaccel.h:
 * qmac8_acc); ACC32=1 restores the original int32_t carrier (11-insn inner
 * loop) so the earlier numbers stay reproducible.                            */
#ifdef ACC32
#define ACC_T int32_t
#define QMAC(s, a, w) qmac8((s), (a), (w))
#define QREQ(s, p)    qreq((s), (p))
#else
#define ACC_T int64_t
#define QMAC(s, a, w) qmac8_acc((s), (a), (w))
#define QREQ(s, p)    qreq_acc((s), (p))
#endif

#define GEMM_QMAC8(NAME, REQ0, REQ1)                                          \
static void NAME(const layer_t *L, int8_t *out)                               \
{                                                                             \
    const int M = L->M, N = L->N, Kp = L->Kp;                                 \
    for (int m = 0; m < M; m++) {                                             \
        const int8_t *a = L->a + (long)m * Kp;                                \
        for (int n = 0; n < N; n += 2) {                                      \
            const int8_t *w0 = L->w + (long)n * Kp;                           \
            const int8_t *w1 = w0 + Kp;                                       \
            ACC_T s0 = L->bias[n], s1 = L->bias[n + 1];                       \
            for (int k = 0; k < Kp; k += 8) {                                 \
                uint64_t pa = LD64(a + k);                                    \
                s0 = QMAC(s0, pa, LD64(w0 + k));                              \
                s1 = QMAC(s1, pa, LD64(w1 + k));                              \
            }                                                                 \
            out[(long)m * N + n]     = REQ0;                                  \
            out[(long)m * N + n + 1] = REQ1;                                  \
        }                                                                     \
    }                                                                         \
}

GEMM_QMAC8(gemm_x0, req_sw(L, n, (int32_t)s0), req_sw(L, n + 1, (int32_t)s1))
GEMM_QMAC8(gemm_x1, QREQ(s0, L->packed[n]), QREQ(s1, L->packed[n + 1]))

/* ---- X1T: X1 with the loop nest re-ordered for the data cache -------------
 * X1 walks the WHOLE weight matrix (N*Kp bytes) once per output row m.  When
 * N*Kp exceeds the D-cache (op05: 36 KB vs CV-Wally's 16 KB) that is a cyclic
 * stream, which LRU/PLRU turn into ~100% misses.  X1T processes a block of `mb`
 * rows of A against each channel pair, so the weights are fetched once per
 * block instead of once per row and the A block (mb*Kp <= MTILE_BYTES) stays
 * resident.  Same arithmetic and result bytes; only the iteration order changes
 * (it also hoists the per-channel qreq operand and bias out of the m loop, so
 * it retires ~1% fewer instructions than X1 - measured, see model_summary.md).
 * Layers with Kp > MTILE_BYTES/2 (Kp = 4608: op29, op32, op33) get mb = 1, i.e.
 * no reuse: they stay weight-streaming (documented limitation).                */
#ifndef MTILE_BYTES
#define MTILE_BYTES 8192
#endif
static void gemm_x1t(const layer_t *L, int8_t *out)
{
    const int M = L->M, N = L->N, Kp = L->Kp;
    /* Block only when the weights do not fit in a tile.  For N*Kp <= MTILE_BYTES
     * (only op14: 128 x 64 B = 8 KB) the X1 order already keeps W resident, and a
     * block of A would just compete with W for the same sets (measured with an
     * unconditional block: op14 0.24% -> 5.3% miss rate at 16 KB). */
    int mb = ((long)N * Kp > MTILE_BYTES) ? MTILE_BYTES / Kp : 1;
    if (mb < 1) mb = 1;
    for (int m0 = 0; m0 < M; m0 += mb) {
        const int m1 = (m0 + mb < M) ? m0 + mb : M;
        for (int n = 0; n < N; n += 2) {
            const int8_t *w0 = L->w + (long)n * Kp;
            const int8_t *w1 = w0 + Kp;
            const uint64_t p0 = L->packed[n], p1 = L->packed[n + 1];
            const int32_t b0 = L->bias[n], b1 = L->bias[n + 1];
            for (int m = m0; m < m1; m++) {
                const int8_t *a = L->a + (long)m * Kp;
                ACC_T s0 = b0, s1 = b1;
                for (int k = 0; k < Kp; k += 8) {
                    uint64_t pa = LD64(a + k);
                    s0 = QMAC(s0, pa, LD64(w0 + k));
                    s1 = QMAC(s1, pa, LD64(w1 + k));
                }
                out[(long)m * N + n]     = QREQ(s0, p0);
                out[(long)m * N + n + 1] = QREQ(s1, p1);
            }
        }
    }
}

/* ---- X2 / X2T: X1 / X1T with the post-increment load `ldpi` ----------------
 * The X1 inner loop is 3 ld + 3 addi (pointer bumps) + 2 qmac8 + 1 bne.  ldpi
 * fuses each load with its pointer bump, leaving 3 ldpi + 2 qmac8 + 1 bne = 6
 * instructions for the same 16 MACs.  Loop structure, results and memory access
 * order are identical to X1 / X1T, so X1 -> X2 isolates ldpi (incremental
 * evaluation, as qmac8 and qreq were).  Kp is a multiple of 8 and >= 8, so the
 * do-while body runs at least once.                                           */
#define INNER_LDPI(s0, s1, a, w0, w1, Kp)                                     \
    do {                                                                      \
        const int8_t *pa_ = (a), *pw0_ = (w0), *pw1_ = (w1);                  \
        const int8_t *end_ = pa_ + (Kp);                                      \
        do {                                                                  \
            uint64_t x_ = ldpi8(&pa_);                                        \
            s0 = QMAC(s0, x_, ldpi8(&pw0_));                                  \
            s1 = QMAC(s1, x_, ldpi8(&pw1_));                                  \
        } while (pa_ != end_);                                                \
    } while (0)

static void gemm_x2(const layer_t *L, int8_t *out)
{
    const int M = L->M, N = L->N, Kp = L->Kp;
    for (int m = 0; m < M; m++) {
        const int8_t *a = L->a + (long)m * Kp;
        for (int n = 0; n < N; n += 2) {
            const int8_t *w0 = L->w + (long)n * Kp;
            const int8_t *w1 = w0 + Kp;
            ACC_T s0 = L->bias[n], s1 = L->bias[n + 1];
            INNER_LDPI(s0, s1, a, w0, w1, Kp);
            out[(long)m * N + n]     = QREQ(s0, L->packed[n]);
            out[(long)m * N + n + 1] = QREQ(s1, L->packed[n + 1]);
        }
    }
}

static void gemm_x2t(const layer_t *L, int8_t *out)
{
    const int M = L->M, N = L->N, Kp = L->Kp;
    int mb = ((long)N * Kp > MTILE_BYTES) ? MTILE_BYTES / Kp : 1;
    if (mb < 1) mb = 1;
    for (int m0 = 0; m0 < M; m0 += mb) {
        const int m1 = (m0 + mb < M) ? m0 + mb : M;
        for (int n = 0; n < N; n += 2) {
            const int8_t *w0 = L->w + (long)n * Kp;
            const int8_t *w1 = w0 + Kp;
            const uint64_t p0 = L->packed[n], p1 = L->packed[n + 1];
            const int32_t b0 = L->bias[n], b1 = L->bias[n + 1];
            for (int m = m0; m < m1; m++) {
                ACC_T s0 = b0, s1 = b1;
                INNER_LDPI(s0, s1, L->a + (long)m * Kp, w0, w1, Kp);
                out[(long)m * N + n]     = QREQ(s0, p0);
                out[(long)m * N + n + 1] = QREQ(s1, p1);
            }
        }
    }
}

/* -DCFG_ONLY=<n> compiles just that configuration (used to measure per-config
 * code size); by default all four are linked and chosen at run time. */
void run_layer(const layer_t *L, int cfg, int8_t *out)
{
#ifdef CFG_ONLY
    cfg = CFG_ONLY;
#endif
    switch (cfg) {
    case CFG_B0: gemm_b0(L, out); break;
    case CFG_B1: gemm_b1(L, out); break;
    case CFG_X0: gemm_x0(L, out); break;
    case CFG_X1T: gemm_x1t(L, out); break;
    case CFG_X2:  gemm_x2(L, out); break;
    case CFG_X2T: gemm_x2t(L, out); break;
    default:     gemm_x1(L, out); break;
    }
}
