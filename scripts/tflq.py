"""TFLite integer requantization arithmetic in numpy (int64), bit-exact.

Mirrors src/tfl_requant.h.  Reference: tensorflow/lite/kernels/internal/common.h
(MultiplyByQuantizedMultiplier) and gemmlowp fixedpoint.
"""
import math
import numpy as np


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul(a, b) for int64 array a, scalar b."""
    ab = a.astype(np.int64) * np.int64(b)
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    t = ab + nudge
    return np.where(t >= 0, t >> 31, -((-t) >> 31)).astype(np.int64)


def rdbp(x, e):
    """RoundingDivideByPOT(x, e), 0 <= e <= 31."""
    mask = (np.int64(1) << e) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> e) + (rem > thr)


def mulq_channels(acc, mult, shift):
    """acc [M,N] int64; mult/shift [N]: per-channel MultiplyByQuantizedMultiplier."""
    out = np.empty_like(acc)
    for c in range(acc.shape[1]):
        ls = max(int(shift[c]), 0)
        rs = max(-int(shift[c]), 0)
        out[:, c] = rdbp(srdhm(acc[:, c] << ls, int(mult[c])), rs)
    return out


def quantize_multiplier(d):
    """double -> (Q31 multiplier, shift), same as TFLite QuantizeMultiplier."""
    if d == 0.0:
        return 0, 0
    q, sh = math.frexp(d)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        sh += 1
    if sh < -31:
        return 0, 0
    return qf, sh


def pack_params(m, sh, zp, amin, amax):
    """64-bit rs2 operand of qreq (see src/nnaccel.h)."""
    return ((m & 0xFFFFFFFF) | ((sh & 0xFF) << 32) | ((zp & 0xFF) << 40) |
            ((amin & 0xFF) << 48) | ((amax & 0xFF) << 56))


def act_range(act, scale, zp):
    lo, hi = -128, 127
    if act == "RELU":
        lo = max(lo, zp)
    elif act == "RELU6":
        lo = max(lo, zp)
        hi = min(hi, zp + int(round(6.0 / scale)))
    elif act == "RELU_N1_TO_1":
        lo = max(lo, zp + int(round(-1.0 / scale)))
        hi = min(hi, zp + int(round(1.0 / scale)))
    return lo, hi
