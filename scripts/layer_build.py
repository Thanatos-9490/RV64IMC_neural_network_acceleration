"""Build one exported layer (blobs + metadata) from a W8A8 .tflite.

"""
import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as S

import tflq

RES = tf.lite.experimental.OpResolverType.BUILTIN_REF
BO = {v: k for k, v in S.BuiltinOperator.__dict__.items()
      if not k.startswith("_") and isinstance(v, int)}
ACT = {0: "NONE", 1: "RELU", 2: "RELU_N1_TO_1", 3: "RELU6"}


def truncated(model_bytes, k):
    m = S.ModelT.InitFromObj(S.Model.GetRootAsModel(model_bytes, 0))
    sg = m.subgraphs[0]
    sg.operators = sg.operators[:k + 1]
    sg.outputs = np.array([sg.operators[-1].outputs[0]], dtype=np.int32)
    f = flatbuffers.Builder(1024)
    f.Finish(m.Pack(f), b"TFL3")
    return bytes(f.Output())


def kind_of(m, op):
    oc = m.operatorCodes[op.opcodeIndex]
    return BO[max(oc.builtinCode, oc.deprecatedBuiltinCode)]


def _im2col(A, W, opt, z_i, Y):
    """A [1,H,W,C] int8, W [N,kh,kw,C] -> Ain [M,kh*kw*C] (tap order ky,kx,c)."""
    N, kh, kw, C = W.shape
    Hh, Ww = A.shape[1], A.shape[2]
    st = opt.strideH
    if opt.padding == 0:                                    # SAME
        oh, ow = -(-Hh // st), -(-Ww // st)
        ph = max((oh - 1) * st + kh - Hh, 0)
        pw = max((ow - 1) * st + kw - Ww, 0)
        A = np.pad(A, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2),
                       (0, 0)), constant_values=z_i)
    Hp, Wp = A.shape[1], A.shape[2]
    oh, ow = (Hp - kh) // st + 1, (Wp - kw) // st + 1
    assert Y.shape == (1, oh, ow, N), (Y.shape, (oh, ow, N))
    Ain = np.empty((oh * ow, kh * kw * C), np.int8)
    for yy in range(oh):
        for xx in range(ow):
            Ain[yy * ow + xx] = A[0, yy * st:yy * st + kh,
                                  xx * st:xx * st + kw, :].reshape(-1)
    return Ain, dict(kh=kh, kw=kw, stride=st, oh=oh, ow=ow, Cin=C)


def _pad_k(X, Kp):
    out = np.zeros((X.shape[0], Kp), np.int8)
    out[:, :X.shape[1]] = X
    return out


def build(label, k, kind, m, model_bytes, x):
    sg = m.subgraphs[0]
    op = sg.operators[k]
    ti, tw, tb, to = op.inputs[0], op.inputs[1], op.inputs[2], op.outputs[0]
    it = tf.lite.Interpreter(model_content=truncated(model_bytes, k),
                             experimental_preserve_all_tensors=True,
                             experimental_op_resolver_type=RES)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], x)
    it.invoke()
    Y = it.get_tensor(it.get_output_details()[0]["index"])
    A, W = it.get_tensor(ti), it.get_tensor(tw)
    B = it.get_tensor(tb).astype(np.int64)
    td = {t["index"]: t for t in it.get_tensor_details()}
    qi, qw, qo = (td[i]["quantization_parameters"] for i in (ti, tw, to))
    s_i, z_i = float(qi["scales"][0]), int(qi["zero_points"][0])
    s_o, z_o = float(qo["scales"][0]), int(qo["zero_points"][0])
    sw = np.asarray(qw["scales"], np.float64)
    sw = np.full(W.shape[0], sw[0]) if sw.size == 1 else sw
    opt = op.builtinOptions
    act = ACT[getattr(opt, "fusedActivationFunction", 0)]
    amin, amax = tflq.act_range(act, s_o, z_o)

    geo = {}
    if kind == "CONV_2D":
        Ain, geo = _im2col(A, W, opt, z_i, Y)
        Wm = np.ascontiguousarray(W.reshape(W.shape[0], -1))
    else:                                                   # FULLY_CONNECTED
        Ain, Wm = A.reshape(1, -1), np.ascontiguousarray(W)
    M, K = Ain.shape
    N = Wm.shape[0]
    Kp = (K + 7) // 8 * 8
    Bf = B - z_i * Wm.astype(np.int64).sum(1)
    acc = Ain.astype(np.int64) @ Wm.astype(np.int64).T + Bf
    mu = np.zeros(N, np.int64)
    sh = np.zeros(N, np.int64)
    for c in range(N):
        mu[c], sh[c] = tflq.quantize_multiplier(s_i * float(sw[c]) / s_o)
    gold = np.clip(tflq.mulq_channels(acc, mu, sh) + z_o, amin, amax)
    gold = gold.astype(np.int8).reshape(Y.shape)

    e = dict(name=label, kind=kind, op_index=k, M=M, N=N, K=K, Kp=Kp,
             act=act, in_zp=z_i, out_zp=z_o, act_min=amin, act_max=amax,
             per_channel=bool(len(qw["scales"]) > 1), macs=M * N * K)
    e.update(geo)
    e["numpy_matches_tflite"] = bool((gold == Y).all())
    e["acc_absmax"] = int(np.abs(acc).max())
    packed = np.array([tflq.pack_params(int(mu[c]), int(sh[c]), z_o, amin, amax)
                       for c in range(N)], np.uint64)
    blobs = dict(a=_pad_k(Ain, Kp), w=_pad_k(Wm.astype(np.int8), Kp),
                 bias=Bf.astype(np.int32), mult=mu.astype(np.int32),
                 shift=sh.astype(np.int8), packed=packed, y=Y.astype(np.int8))
    return e, blobs
