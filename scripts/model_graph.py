"""Whole-graph extraction of the W8A8 ResNet-18 TFLite model (37 ops).

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
import tflq                         # noqa: E402

ALIGN = 64
LSHIFT = 20            # TFLite ADD, 8-bit: left_shift = 20


def op_hash(arr):
    """(sum, position-weighted sum) of the int8 bytes, mod 2^64.  Detects any
    changed byte and (almost) any permutation; cheap to recompute on target."""
    u = (arr.reshape(-1).astype(np.int64) & 0xFF).astype(np.uint64)
    pos = np.arange(1, u.size + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(u.sum(dtype=np.uint64)), int(((u + np.uint64(1)) * pos).sum(dtype=np.uint64))


def add_params(s1, z1, s2, z2, so, zo, act):
    """TFLite ADD int8 (PrepareGeneralSub/Add): both inputs rescaled to 2*max(scale)."""
    twice = 2.0 * max(s1, s2)
    m1, sh1 = tflq.quantize_multiplier(s1 / twice)
    m2, sh2 = tflq.quantize_multiplier(s2 / twice)
    mo, sho = tflq.quantize_multiplier(twice / ((1 << LSHIFT) * so))
    lo, hi = tflq.act_range(act, so, zo)
    return dict(add_in1_off=-z1, add_in2_off=-z2, add_lshift=LSHIFT, add_m1=m1,
                add_s1=sh1, add_m2=m2, add_s2=sh2, add_mo=mo, add_so=sho,
                out_zp=zo, act_min=lo, act_max=hi)


def plan_arena(sizes, defs, last):
    """First-fit placement of tensors with overlapping lifetimes."""
    rnd = lambda n: (n + ALIGN - 1) // ALIGN * ALIGN
    placed, off = [], {}
    for t in sorted(sizes, key=lambda t: defs[t]):
        busy = sorted((o, o + rnd(sizes[u])) for u, o in placed
                      if not (last[u] < defs[t] or last[t] < defs[u]))
        cur = 0
        for lo, hi in busy:
            if cur + rnd(sizes[t]) <= lo:
                break
            cur = max(cur, hi)
        off[t] = cur
        placed.append((t, cur))
    return off, max(o + rnd(sizes[t]) for t, o in off.items())
