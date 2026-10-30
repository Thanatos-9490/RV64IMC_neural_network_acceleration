#!/usr/bin/env python3
"""Export representative ResNet-18 W8A8 layers for bare-metal bit-exact checks.

Run inside `conda activate aca_term_prj` (needs TensorFlow).  Writes gen/
(committed, so `make all` does not need TensorFlow).  See layer_build.py for
the exact-integer lowering and the ground-truth method.

Exports ALL 20 CONV_2D + 1 FULLY_CONNECTED layers of ResNet-18 (100% of the
MACs), each with its real input activation from one image.  Layer names are
"op<NN>" = index of the op in the TFLite graph.
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import export_c                      # noqa: E402
import layer_build as LB             # noqa: E402
import quantize_model as Q           # noqa: E402

MODEL = os.path.join(ROOT, "models", "resnet18_w8a8.tflite")
OUT = os.path.join(ROOT, "gen")

def wanted(m):
    """every CONV_2D / FULLY_CONNECTED op of the graph, in execution order"""
    out = []
    for k, op in enumerate(m.subgraphs[0].operators):
        kind = LB.kind_of(m, op)
        if kind in ("CONV_2D", "FULLY_CONNECTED"):
            out.append(("op%02d" % k, k, kind))
    return out


def main():
    b = open(MODEL, "rb").read()
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(b, 0))
    it0 = tf.lite.Interpreter(model_content=b, experimental_op_resolver_type=LB.RES)
    it0.allocate_tensors()
    s0, z0 = it0.get_input_details()[0]["quantization"]
    img = Q.load_u8(Q.sample("val", 1, 3)[0][0])
    x = np.clip(np.round(Q.to_input(img, "imagenet") / s0) + z0, -128, 127).astype(np.int8)
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):                       # drop stale exports
        if f.endswith(".bin") or f == "layers.json":
            os.remove(os.path.join(OUT, f))
    meta = []
    for label, k, kind in wanted(m):
        assert LB.kind_of(m, m.subgraphs[0].operators[k]) == kind, (label, k)
        e, blobs = LB.build(label, k, kind, m, b, x)
        print("%-9s op#%-3d %-16s M=%-6d N=%-4d K=%-5d Kp=%-5d MACs=%-10d act=%-5s "
              "acc_absmax=%-9d numpy==TFLite: %s"
              % (label, k, kind, e["M"], e["N"], e["K"], e["Kp"], e["macs"], e["act"],
                 e["acc_absmax"], e["numpy_matches_tflite"]), flush=True)
        for key, v in blobs.items():
            open(os.path.join(OUT, f"{label}_{key}.bin"), "wb").write(v.tobytes())
        meta.append(e)
    json.dump(meta, open(os.path.join(OUT, "layers.json"), "w"), indent=2)
    export_c.emit(meta, OUT)
    return 0 if meta and all(e["numpy_matches_tflite"] for e in meta) else 1


if __name__ == "__main__":
    sys.exit(main())
