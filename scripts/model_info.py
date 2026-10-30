#!/usr/bin/env python3
"""Workload characterization of a .tflite model (op histogram + MACs/layer).

Run inside `conda activate aca_term_prj` (uses TF's flatbuffer schema).

    python scripts/model_info.py models/resnet18.tflite [--csv out.csv]
"""
import argparse
import collections
import sys

from tensorflow.lite.python import schema_py_generated as S

BO = {v: k for k, v in S.BuiltinOperator.__dict__.items()
      if not k.startswith("_") and isinstance(v, int)}
ACT = {0: "NONE", 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6"}
PAD = {0: "SAME", 1: "VALID"}


def load(path):
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(open(path, "rb").read(), 0))
    return m, m.subgraphs[0]


def opname(m, op):
    oc = m.operatorCodes[op.opcodeIndex]
    return BO[max(oc.builtinCode, oc.deprecatedBuiltinCode)]


def layers(path):
    """Yield one dict per operator with shapes, options and MAC count."""
    m, sg = load(path)
    out = []
    for i, op in enumerate(sg.operators):
        name = opname(m, op)
        ins = [sg.tensors[t] for t in op.inputs if t >= 0]
        o = sg.tensors[op.outputs[0]]
        d = dict(idx=i, op=name, in_shape=list(ins[0].shape),
                 out_shape=list(o.shape), macs=0, k="", stride="", pad="",
                 act="", params=0)
        opt = op.builtinOptions
        if name in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            w = list(ins[1].shape)                       # [OC,KH,KW,C]
            n, oh, ow, oc = o.shape
            cin = w[3] if name == "CONV_2D" else 1
            d.update(macs=n * oh * ow * oc * w[1] * w[2] * cin,
                     k="%dx%d" % (w[1], w[2]), stride=opt.strideH,
                     pad=PAD[opt.padding], act=ACT[opt.fusedActivationFunction],
                     params=int(oc * w[1] * w[2] * cin))
        elif name == "FULLY_CONNECTED":
            w = list(ins[1].shape)
            d.update(macs=w[0] * w[1], params=w[0] * w[1],
                     act=ACT[opt.fusedActivationFunction])
        elif name in ("MAX_POOL_2D", "AVERAGE_POOL_2D"):
            d.update(k="%dx%d" % (opt.filterHeight, opt.filterWidth),
                     stride=opt.strideH, pad=PAD[opt.padding],
                     act=ACT[opt.fusedActivationFunction])
        elif name == "ADD":
            d.update(act=ACT[opt.fusedActivationFunction])
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--csv")
    a = ap.parse_args()
    L = layers(a.model)
    tot = sum(l["macs"] for l in L)
    print("%-4s %-18s %-18s %-18s %-4s %-3s %-5s %-5s %12s %6s" %
          ("#", "op", "in", "out", "k", "s", "pad", "act", "MACs", "%"))
    for l in L:
        print("%-4d %-18s %-18s %-18s %-4s %-3s %-5s %-5s %12d %5.1f%%" %
              (l["idx"], l["op"], l["in_shape"], l["out_shape"], l["k"],
               l["stride"], l["pad"], l["act"], l["macs"],
               100.0 * l["macs"] / tot if tot else 0))
    h = collections.Counter(l["op"] for l in L)
    byop = collections.Counter()
    for l in L:
        byop[l["op"]] += l["macs"]
    print("\nop histogram:", dict(h))
    print("total MACs: %.1f M (%.3f GMAC)   weights: %.2f M" %
          (tot / 1e6, tot / 1e9, sum(l["params"] for l in L) / 1e6))
    for k, v in byop.most_common():
        if v:
            print("  %-18s %12.1f M  (%.1f%%)" % (k, v / 1e6, 100.0 * v / tot))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(L[0].keys()))
            w.writeheader()
            w.writerows(L)


if __name__ == "__main__":
    sys.exit(main())
