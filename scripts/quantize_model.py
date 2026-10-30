#!/usr/bin/env python3
"""Post-training quantization of MobileNetV3-Large (FP32 .tflite) + accuracy.
"""
import json
import os
import random
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.lite.python.optimize import calibrator as _calib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "imagenette2-160")
MODELS = os.path.join(ROOT, "models")
FP32 = os.path.join(MODELS, "mobilenet_v3_large.tflite")

SYN2IDX = {"n01440764": 0, "n02102040": 217, "n02979186": 482,
           "n03000684": 491, "n03028079": 497, "n03394916": 566,
           "n03417042": 569, "n03425413": 571, "n03445777": 574,
           "n03888257": 701}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
N_CALIB, N_EVAL = 20, 50          # images per class


def sample(split, per_class, seed):
    rng, out = random.Random(seed), []
    for syn, idx in sorted(SYN2IDX.items()):
        d = os.path.join(DATA, split, syn)
        fs = sorted(os.listdir(d))
        rng.shuffle(fs)
        out += [(os.path.join(d, f), idx) for f in fs[:per_class]]
    return out


def load_u8(path):
    """Resize shorter side to 256, center-crop 224 -> uint8 HWC."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256.0 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224)), np.uint8)


def to_input(u8, norm):
    x = u8.astype(np.float32) / 255.0
    if norm == "imagenet":
        x = (x - MEAN) / STD
    elif norm == "pm1":
        x = x * 2.0 - 1.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])   # NCHW


def evaluate(model_bytes, imgs, labels, norm):
    it = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = inp["quantization"]
    hit = 0
    for u8, y in zip(imgs, labels):
        x = to_input(u8, norm)
        if inp["dtype"] != np.float32:
            info = np.iinfo(inp["dtype"])
            x = np.clip(np.round(x / scale) + zp, info.min, info.max)
            x = x.astype(inp["dtype"])
        it.set_tensor(inp["index"], x)
        it.invoke()
        hit += int(np.argmax(it.get_tensor(out["index"])[0]) == y)
    return hit / len(labels)


def quantize(fp_bytes, calib_u8, norm, mode, allow_float=False):
    def gen():
        for u8 in calib_u8:
            yield [to_input(u8, norm)]
    c = _calib.Calibrator(fp_bytes)
    if mode == "w8a8":
        return c.calibrate_and_quantize(gen, tf.int8, tf.int8, allow_float,
                                        tf.int8, tf.int32)
    return c.calibrate_and_quantize(gen, tf.int16, tf.int16, allow_float,
                                    tf.int16, tf.int64)


# ---------------------------------------------------------------------------
# Repair of a numerically collapsed Squeeze-Excite block in the FP32 checkpoint
# ---------------------------------------------------------------------------
def repair_fp32(fp_bytes):
    """features_4_block_2 (SE) has fc1/fc2 weights ~1e-8, so TFLite PTQ derives
    a denormal activation scale (4.6e-41) and inf weight/bias scales and aborts.
    The SE gate there is a constant, hardsigmoid(fc2_bias).  We keep that
    function (to ~1e-4 on the pre-activation) but make the tensors
    quantizable:  fc1: w[c,0]=EPS, b[0]=1  ->  relu out = (1,0,0,...)
                  fc2: w[c,0]=EPS, b unchanged."""
    from tensorflow.lite.python import schema_py_generated as S
    import flatbuffers
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(bytearray(fp_bytes), 0))
    sg = m.subgraphs[0]
    byname = {(t.name.decode() if isinstance(t.name, bytes) else t.name): t
              for t in sg.tensors}
    EPS = np.float32(1e-4)
    pre = "g0::p_features_4_block_2_"

    def setbuf(t, arr):
        m.buffers[t.buffer].data = np.frombuffer(
            np.ascontiguousarray(arr, np.float32).tobytes(), np.uint8)

    w1, b1 = byname[pre + "fc1_weight"], byname[pre + "fc1_bias"]
    w2 = byname[pre + "fc2_weight"]
    a = np.zeros(w1.shape, np.float32); a[:, 0, 0, 0] = EPS; setbuf(w1, a)
    b = np.zeros(b1.shape, np.float32); b[0] = 1.0;          setbuf(b1, b)
    a = np.zeros(w2.shape, np.float32); a[:, 0, 0, 0] = EPS; setbuf(w2, a)
    f = flatbuffers.Builder(1024)
    f.Finish(m.Pack(f), b"TFL3")
    return bytes(f.Output())


def main():
    """usage: quantize_model.py [mobilenet_v3_large|resnet18]"""
    name = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    fp = open(os.path.join(MODELS, name + ".tflite"), "rb").read()
    # only MobileNetV3's checkpoint has the collapsed SE block
    fixed = repair_fp32(fp) if name == "mobilenet_v3_large" else fp
    ev = sample("val", N_EVAL, 11)
    ev_u8 = [load_u8(p) for p, _ in ev]
    ev_y = [y for _, y in ev]
    cal_u8 = [load_u8(p) for p, _ in sample("train", 5, 0)]
    rep = {"model": name, "eval_images": len(ev), "calib_images": len(cal_u8),
           "preprocess": "resize256+centercrop224, ImageNet mean/std"}
    rep["fp32_top1"] = evaluate(fixed, ev_u8, ev_y, "imagenet")
    for mode in ("w8a8", "w8a16"):
        try:
            q = quantize(fixed, cal_u8, "imagenet", mode)
            open(os.path.join(MODELS, "%s_%s.tflite" % (name, mode)), "wb").write(q)
            rep[mode + "_top1"] = evaluate(q, ev_u8, ev_y, "imagenet")
            rep[mode + "_bytes"] = len(q)
        except Exception as e:
            rep[mode + "_error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    os.makedirs(os.path.join(ROOT, "results", "processed"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, "results", "processed",
                                     "quant_report_%s.json" % name), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
