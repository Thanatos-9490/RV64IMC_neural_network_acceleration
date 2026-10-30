/* ---------------------------------------------------------------------------
 * run_model.c - FULL ResNet-18 W8A8 inference, all 37 ops chained, bare-metal.
 * ------------------------------------------------------------------------- */
#include <stdint.h>
#include "layer_kernels.h"
#include "model_ops.h"
#include "htif.h"
#include "model.h"

#ifndef CFG
#define CFG CFG_X1
#endif
#ifndef NIMG
#define NIMG M_NUM_IMAGES
#endif

#define PH_IGNORE 63          /* phase id the cache simulator drops (ext/cachesweep.cc) */

static int8_t arena[M_ARENA_BYTES] __attribute__((aligned(64)));
static int8_t col[M_COL_BYTES] __attribute__((aligned(64)));
volatile uint8_t phase_mark[64] __attribute__((aligned(64)));

static inline const int8_t *tensor_ro(int id, const int8_t *img)
{
    return id == 0 ? img : arena + M_TENSOR_OFF[id];
}

static inline int8_t *tensor_rw(int id)
{
    return arena + M_TENSOR_OFF[id];
}

/* position-weighted checksum, identical to scripts/model_graph.py:op_hash */
__attribute__((noinline))
static int check_op(const mop_t *o)
{
    const int8_t *p = tensor_ro(o->out, 0);
    const long n = M_TENSOR_BYTES[o->out];
    uint64_t s = 0, h = 0;
    for (long i = 0; i < n; i++) {
        uint64_t u = (uint8_t)p[i];
        s += u;
        h += (u + 1) * (uint64_t)(i + 1);
    }
    return !(s == o->h_sum && h == o->h_pos);
}

__attribute__((noinline))
static void exec_op(const mop_t *o, const int8_t *img)
{
    const int8_t *in = tensor_ro(o->in0, img);
    int8_t *out = tensor_rw(o->out);
    switch (o->kind) {
    case MK_TRANSPOSE:
        op_transpose(in, out, o->ic, o->ih, o->iw);
        break;
    case MK_PAD:
        op_pad(in, out, o->ih, o->iw, o->ic, o->oh, o->ow, o->pad_t, o->pad_l,
               o->out_zp);
        break;
    case MK_CONV: {
        layer_t L = M_LAYERS[o->layer];
        op_im2col(in, col, o->ih, o->iw, o->ic, o->kh, o->kw, o->stride,
                  o->pad_t, o->pad_l, o->oh, o->ow, o->in_zp, L.Kp);
        L.a = col;
        run_layer(&L, CFG, out);
        break;
    }
    case MK_FC: {
        layer_t L = M_LAYERS[o->layer];
        L.a = in;                                  /* M = 1, K = Kp = 512 */
        run_layer(&L, CFG, out);
        break;
    }
    case MK_MAXPOOL:
        op_maxpool(in, out, o->ih, o->iw, o->ic, o->kh, o->stride, o->oh, o->ow,
                   o->act_min, o->act_max);
        break;
    case MK_ADD:
        op_add(in, tensor_ro(o->in1, img), out, M_TENSOR_BYTES[o->out],
               o->add_in1_off, o->add_in2_off, o->add_lshift, o->add_m1,
               o->add_s1, o->add_m2, o->add_s2, o->add_mo, o->add_so, o->out_zp,
               o->act_min, o->act_max);
        break;
    default:                                        /* MK_AVGPOOL */
        op_avgpool(in, out, o->ih, o->iw, o->ic, o->act_min, o->act_max);
        break;
    }
}

static int argmax(const int8_t *v, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[b])
            b = i;
    return b;
}

static void put(const char *tag, uint64_t a, uint64_t b)
{
    uint64_t v[2] = {a, b};
    htif_line(tag, v, 2);
}

int main(void)
{
    long bad_ops = 0, bad_logits = 0, top1 = 0, tf_top1 = 0;
#ifdef REPORT_INSTRET
    uint64_t inferred = 0;          /* instret between the two reads around exec_op() */
    uint64_t overhead = 0;          /* cost of the read pair itself, measured below */
    {   uint64_t a = rd_instret(), b = rd_instret();
        overhead = b - a;           /* instructions retired between two back-to-back reads */
    }
#endif
    for (int im = 0; im < NIMG; im++) {
        const int8_t *img = M_IMAGES + (long)im * M_IMG_BYTES;
        for (int k = 0; k < M_NUM_OPS; k++) {
            phase_mark[k] = 1;                      /* phase k = inference of op k */
#ifdef REPORT_INSTRET
            uint64_t t0 = rd_instret();
#endif
            exec_op(&M_OPS[k], img);
#ifdef REPORT_INSTRET
            inferred += rd_instret() - t0;
#endif
            phase_mark[PH_IGNORE] = 1;              /* verification is not simulated */
            if (im == 0 && check_op(&M_OPS[k])) {
                bad_ops++;
                put("MISMATCH_OP", k, M_OPS[k].kind);
            }
        }
        phase_mark[PH_IGNORE] = 1;
        const int8_t *lg = tensor_ro(M_OUT_TENSOR, 0);
        const int8_t *gl = M_LOGITS + (long)im * M_NUM_CLASSES;
        long d = 0;
        for (int i = 0; i < M_NUM_CLASSES; i++)
            d += (lg[i] != gl[i]);
        bad_logits += d;
        top1 += (argmax(lg, M_NUM_CLASSES) == M_LABELS[im]);
        tf_top1 += (argmax(gl, M_NUM_CLASSES) == M_LABELS[im]);
        put("IMAGE", im, d);
    }
#ifdef REPORT_INSTRET
    {   /* `inferred` counts, per op, what retires between the two reads: exec_op()
         * plus the csr read itself.  `overhead` (back-to-back reads) is that CSR
         * read's own contribution (the first read retires inside the window). */
        uint64_t v[1] = {inferred - (uint64_t)NIMG * M_NUM_OPS * overhead};
        htif_line("INSTRET", v, 1);
    }
#endif
    put("SUMMARY_bad_ops_bad_logits", bad_ops, bad_logits);
    put("SUMMARY_top1_tflite_top1", top1, tf_top1);
    long bad = bad_ops + bad_logits;
    return bad ? (int)(bad > 120 ? 120 : bad) : 0;
}
