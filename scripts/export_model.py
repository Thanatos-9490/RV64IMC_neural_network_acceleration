#!/usr/bin/env python3
"""Export the WHOLE ResNet-18 W8A8 graph (37 ops) for bare-metal chained inference.

Run inside `conda activate aca_term_prj` (needs TensorFlow).  Writes gen/model.h,
gen/model.S, gen/model_*.bin, gen/model.json (committed, so `make` needs no TF).

"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import layer_build as LB            # noqa: E402
import model_graph as G             # noqa: E402
import quantize_model as Q          # noqa: E402

MODEL = os.path.join(ROOT, "models", "resnet18_w8a8.tflite")
GEN = os.path.join(ROOT, "gen")
KIND = {"TRANSPOSE": "MK_TRANSPOSE", "PAD": "MK_PAD", "CONV_2D": "MK_CONV",
        "MAX_POOL_2D": "MK_MAXPOOL", "ADD": "MK_ADD",
        "AVERAGE_POOL_2D": "MK_AVGPOOL", "FULLY_CONNECTED": "MK_FC"}
BLOBS = ["w", "bias", "mult", "shift", "packed"]
CT = {"w": "int8_t", "bias": "int32_t", "mult": "int32_t", "shift": "int8_t",
      "packed": "uint64_t"}


def describe_ops(m, it, td, lay):
    """One dict per graph op with the parameters the runtime needs."""
    sg = m.subgraphs[0]
    qp = lambda i: (float(td[i]["quantization_parameters"]["scales"][0]),
                    int(td[i]["quantization_parameters"]["zero_points"][0]))
    shp = lambda i: [int(v) for v in td[i]["shape"]]
    ops = []
    for k, op in enumerate(sg.operators):
        kind = LB.kind_of(m, op)
        ins = [int(v) for v in op.inputs]
        o = op.builtinOptions
        out = int(op.outputs[0])
        d = dict(k=k, kind=kind, name="op%02d" % k, in0=ins[0], in1=-1, out=out,
                 ih=0, iw=0, ic=0, oh=0, ow=0, oc=0, kh=0, kw=0, stride=1,
                 pad_t=0, pad_l=0, in_zp=0, out_zp=0, act_min=-128, act_max=127,
                 layer=-1, add_in1_off=0, add_in2_off=0, add_lshift=0, add_m1=0,
                 add_s1=0, add_m2=0, add_s2=0, add_mo=0, add_so=0)
        si, zi = qp(ins[0])
        so_, zo = qp(out)
        a, y = shp(ins[0]), shp(out)
        act = LB.ACT[getattr(o, "fusedActivationFunction", 0)] if o is not None else "NONE"
        d.update(in_zp=zi, out_zp=zo)
        if kind == "TRANSPOSE":
            assert it.get_tensor(ins[1]).tolist() == [0, 2, 3, 1]
            d.update(ic=a[1], ih=a[2], iw=a[3], oh=y[1], ow=y[2], oc=y[3])
        elif kind == "PAD":
            p = it.get_tensor(ins[1]).tolist()
            assert p[0] == [0, 0] and p[3] == [0, 0] and zi == zo
            d.update(ih=a[1], iw=a[2], ic=a[3], oh=y[1], ow=y[2], oc=y[3],
                     pad_t=p[1][0], pad_l=p[2][0])
        elif kind in ("CONV_2D", "FULLY_CONNECTED"):
            e = lay[d["name"]]
            conv = kind == "CONV_2D"
            assert (e["M"], e["N"]) == ((y[1] * y[2], y[3]) if conv else (1, y[1]))
            d.update(ih=a[1] if conv else 1, iw=a[2] if conv else 1,
                     ic=a[3] if conv else a[-1], oc=e["N"], oh=e.get("oh", 1),
                     ow=e.get("ow", 1), kh=e.get("kh", 1), kw=e.get("kw", 1),
                     stride=e.get("stride", 1), act_min=e["act_min"],
                     act_max=e["act_max"], out_zp=e["out_zp"], in_zp=e["in_zp"],
                     layer=sum(1 for q in ops if q["layer"] >= 0))
            if conv and o.padding == 0:                              # SAME
                ph = max((d["oh"] - 1) * d["stride"] + d["kh"] - d["ih"], 0)
                pw = max((d["ow"] - 1) * d["stride"] + d["kw"] - d["iw"], 0)
                d.update(pad_t=ph // 2, pad_l=pw // 2)
            d.update(M=e["M"], K=e["K"], Kp=e["Kp"])
        elif kind in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            assert o.padding == 1                                    # VALID
            lo, hi = G.tflq.act_range(act, so_, zo)
            if kind == "AVERAGE_POOL_2D":
                assert (si, zi) == (so_, zo)
            d.update(ih=a[1], iw=a[2], ic=a[3], oh=y[1], ow=y[2], oc=y[3],
                     kh=o.filterHeight, kw=o.filterWidth, stride=o.strideH,
                     act_min=lo, act_max=hi)
        elif kind == "ADD":
            s2, z2 = qp(ins[1])
            d.update(in1=ins[1], ih=a[1], iw=a[2], ic=a[3], oh=y[1], ow=y[2], oc=y[3])
            d.update(G.add_params(si, zi, s2, z2, so_, zo, act))
        else:
            raise SystemExit("unsupported op " + kind)
        ops.append(d)
    return ops


def op_golden(b, k, x):
    """Output tensor of op k for input x (graph truncated at k; see layer_build)."""
    i0 = tf.lite.Interpreter(model_content=LB.truncated(b, k),
                             experimental_op_resolver_type=LB.RES)
    i0.allocate_tensors()
    i0.set_tensor(i0.get_input_details()[0]["index"], x)
    i0.invoke()
    return i0.get_tensor(i0.get_output_details()[0]["index"]).astype(np.int8)


def write_blobs(ops, imgs, logits):
    conv = [d for d in ops if d["layer"] >= 0]
    open(os.path.join(GEN, "model_in.bin"), "wb").write(b"".join(i.tobytes() for i in imgs))
    open(os.path.join(GEN, "model_logits.bin"), "wb").write(logits.tobytes())
    with open(os.path.join(GEN, "model.S"), "w") as f:
        f.write("/* generated by scripts/export_model.py - do not edit */\n"
                "    .section .rodata\n    .balign 8\n")
        for d in conv:
            for s in BLOBS:
                f.write(f"    .globl M_{d['name']}_{s}\nM_{d['name']}_{s}:\n"
                        f"    .incbin \"gen/{d['name']}_{s}.bin\"\n    .balign 8\n")
        for sym, fn in (("M_IMAGES", "model_in"), ("M_LOGITS", "model_logits")):
            f.write(f"    .globl {sym}\n{sym}:\n    .incbin \"gen/{fn}.bin\"\n    .balign 8\n")


HDR_TYPES = (
    "/* generated by scripts/export_model.py - do not edit */\n"
    "#ifndef MODEL_H\n#define MODEL_H\n#include \"layer_t.h\"\n\n"
    "enum { MK_TRANSPOSE, MK_PAD, MK_CONV, MK_MAXPOOL, MK_ADD, MK_AVGPOOL, MK_FC };\n\n"
    "typedef struct {\n    int kind; const char *name;\n"
    "    int in0, in1, out;                 /* tensor ids (0 = model input) */\n"
    "    int ih, iw, ic, oh, ow, oc;\n    int kh, kw, stride;\n"
    "    int pad_t, pad_l;                  /* leading pad (SAME: top/left) */\n"
    "    int in_zp, out_zp, act_min, act_max;\n"
    "    int add_in1_off, add_in2_off, add_lshift;\n"
    "    int add_m1, add_s1, add_m2, add_s2, add_mo, add_so;\n"
    "    int layer;                         /* index in M_LAYERS or -1 */\n"
    "    uint64_t h_sum, h_pos;             /* golden checksums of the op output */\n"
    "} mop_t;\n\n")


def write_header(ops, tid, arena_off, arena_bytes, col_bytes, shp, t_in, t_out,
                 imgs, labels, hashes):
    conv = [d for d in ops if d["layer"] >= 0]
    with open(os.path.join(GEN, "model.h"), "w") as f:
        w = f.write
        w(HDR_TYPES)
        w("#define M_NUM_OPS %d\n#define M_NUM_IMAGES %d\n#define M_NUM_TENSORS %d\n"
          "#define M_ARENA_BYTES %d\n#define M_COL_BYTES %d\n#define M_IMG_BYTES %d\n"
          "#define M_NUM_CLASSES %d\n#define M_OUT_TENSOR %d\n\n"
          % (len(ops), len(imgs), len(tid), arena_bytes, col_bytes,
             int(np.prod(shp(t_in))), int(np.prod(shp(t_out))), tid[t_out]))
        w("extern const int8_t M_IMAGES[], M_LOGITS[];\n")
        for d in conv:
            for s in BLOBS:
                w(f"extern const {CT[s]} M_{d['name']}_{s}[];\n")
        w("\nstatic const int M_LABELS[M_NUM_IMAGES] = {%s};\n\n"
          % ", ".join(str(v) for v in labels))
        offs, by = [-1] * len(tid), [0] * len(tid)
        for t, i in tid.items():
            by[i] = int(np.prod(shp(t)))
            if t != t_in:
                offs[i] = arena_off[t]
        w("/* arena offset of each tensor; -1 = model input (read from M_IMAGES) */\n"
          "static const long M_TENSOR_OFF[M_NUM_TENSORS] = {%s};\n"
          "static const int  M_TENSOR_BYTES[M_NUM_TENSORS] = {%s};\n\n"
          % (", ".join(map(str, offs)), ", ".join(map(str, by))))
        w("/* GEMM-form layers; a/y are NULL (filled / checked at run time) */\n"
          "static const layer_t M_LAYERS[] = {\n")
        for d in conv:
            n = d["name"]
            w('    {"%s", %d,%d,%d,%d, %d,%d,%d, 0, M_%s_w, 0, M_%s_bias, M_%s_mult,\n'
              "     M_%s_shift, M_%s_packed},\n"
              % (n, d["M"], d["oc"], d["K"], d["Kp"], d["out_zp"], d["act_min"],
                 d["act_max"], n, n, n, n, n))
        w("};\n\nstatic const mop_t M_OPS[M_NUM_OPS] = {\n")
        for d in ops:
            hs, hp = hashes[d["k"]]
            w('    {%s, "%s", %d,%d,%d, %d,%d,%d,%d,%d,%d, %d,%d,%d, %d,%d, %d,%d,%d,%d,\n'
              "     %d,%d,%d, %d,%d,%d,%d,%d,%d, %d, %dULL,%dULL},\n"
              % (KIND[d["kind"]], d["name"], tid[d["in0"]],
                 tid[d["in1"]] if d["in1"] >= 0 else -1, tid[d["out"]],
                 d["ih"], d["iw"], d["ic"], d["oh"], d["ow"], d["oc"],
                 d["kh"], d["kw"], d["stride"], d["pad_t"], d["pad_l"],
                 d["in_zp"], d["out_zp"], d["act_min"], d["act_max"],
                 d["add_in1_off"], d["add_in2_off"], d["add_lshift"], d["add_m1"],
                 d["add_s1"], d["add_m2"], d["add_s2"], d["add_mo"], d["add_so"],
                 d["layer"], hs, hp))
        w("};\n#endif\n")



def main():
    b = open(MODEL, "rb").read()
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(b, 0))
    sg = m.subgraphs[0]
    lay = {e["name"]: e for e in json.load(open(os.path.join(GEN, "layers.json")))}
    it = tf.lite.Interpreter(model_content=b, experimental_op_resolver_type=LB.RES,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    shp = lambda i: [int(v) for v in td[i]["shape"]]
    t_in, t_out = int(sg.inputs[0]), int(sg.outputs[0])
    qi = td[t_in]["quantization_parameters"]
    s0, z0 = float(qi["scales"][0]), int(qi["zero_points"][0])

    imgs, labels = [], []                        # one image per Imagenette class
    for path, lab in Q.sample("val", 1, 3):
        x = Q.to_input(Q.load_u8(path), "imagenet")
        imgs.append(np.clip(np.round(x / s0) + z0, -128, 127).astype(np.int8))
        labels.append(lab)

    def full(x):
        i2 = tf.lite.Interpreter(model_content=b, experimental_op_resolver_type=LB.RES)
        i2.allocate_tensors()
        i2.set_tensor(i2.get_input_details()[0]["index"], x)
        i2.invoke()
        return i2.get_tensor(i2.get_output_details()[0]["index"])[0].copy()

    logits = np.stack([full(x) for x in imgs]).astype(np.int8)
    hits = sum(int(np.argmax(l) == y) for l, y in zip(logits, labels))
    print("TFLite (BUILTIN_REF): %d images, top-1 %d/%d" % (len(imgs), hits, len(imgs)))

    ops = describe_ops(m, it, td, lay)
    sizes, defs, last = {}, {}, {}
    for d in ops:
        for t in (d["in0"], d["in1"]):
            if t >= 0 and t != t_in:
                last[t] = max(last.get(t, -1), d["k"])
        sizes[d["out"]] = int(np.prod(shp(d["out"])))
        defs[d["out"]] = d["k"]
        last[d["out"]] = max(last.get(d["out"], -1), d["k"])
    last[t_out] = len(ops)
    arena_off, arena_bytes = G.plan_arena(sizes, defs, last)
    tid = {t_in: 0}
    for t in sorted(sizes, key=lambda t: defs[t]):
        tid[t] = len(tid)
    col_bytes = max(d["M"] * d["Kp"] for d in ops if d["kind"] == "CONV_2D")

    hashes = {}
    for d in ops:
        y = op_golden(b, d["k"], imgs[0])
        assert y.size == sizes[d["out"]]
        hashes[d["k"]] = G.op_hash(y)
        if d["k"] == len(ops) - 1:
            assert (y.reshape(-1) == logits[0]).all(), "chained golden != full graph"
    write_blobs(ops, imgs, logits)
    write_header(ops, tid, arena_off, arena_bytes, col_bytes, shp, t_in, t_out,
                 imgs, labels, hashes)
    json.dump(dict(ops=len(ops), arena_bytes=arena_bytes, col_bytes=col_bytes,
                   images=len(imgs), tflite_top1=int(hits), labels=labels,
                   op_names=[d["name"] for d in ops], op_kinds=[d["kind"] for d in ops],
                   op_macs=[d["M"] * d["oc"] * d["K"] if d["layer"] >= 0 else 0
                            for d in ops]),
              open(os.path.join(GEN, "model.json"), "w"), indent=1)
    print("ops=%d arena=%.2f MB im2col=%.2f MB images=%d"
          % (len(ops), arena_bytes / 2**20, col_bytes / 2**20, len(imgs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

