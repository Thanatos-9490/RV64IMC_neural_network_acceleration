#!/usr/bin/env python3
"""Report figures from the generated result files -> results/figures/*.png

  fig_progress.png  instruction count per config + MAC/insn   (ISA-level, whole model)
  fig_mix.png       instruction mix per config
  fig_ops.png       contribution of operator classes
  fig_cycles.png    ISA-only vs cycle-model speedup (model!, see perf_model.md)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "results", "processed")
F = os.path.join(ROOT, "results", "figures")
os.makedirs(F, exist_ok=True)
MP = json.load(open(os.path.join(P, "model_profile.json")))
CFG = [c for c in ("b0", "b1", "x0", "x1", "x2", "x2t") if c in MP]
NAME = {"b0": "B0", "b1": "B1", "x0": "X0\n+qmac8", "x1": "X1\n+qreq", "x2": "X2\n+ldpi",
        "x2t": "X2T\n+blocking", "x1t": "X1T"}
MACS = 1814073344


def tot(c):
    return sum(sum(m.values()) for m in MP[c]["groups"].values())


def mix(c):
    m = {}
    for g in MP[c]["groups"].values():
        for k, v in g.items():
            m[k] = m.get(k, 0) + v
    return m


def progress():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    t = [tot(c) / 1e9 for c in CFG]
    b = ax[0].bar([NAME[c] for c in CFG], t, color=["#888", "#888", "tab:blue", "tab:blue", "tab:green", "tab:red"][:len(CFG)])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("dynamic instructions (G, log)")
    ax[0].set_title("whole ResNet-18 inference (37 ops)", fontsize=10)
    for r, c in zip(b, CFG):
        ax[0].text(r.get_x() + r.get_width() / 2, r.get_height() * 1.05,
                   "%.2fx" % (tot("b1") / tot(c)), ha="center", fontsize=8)
    ax[0].text(0.02, 0.04, "label = speedup vs B1", transform=ax[0].transAxes, fontsize=8)
    ax[1].bar([NAME[c] for c in CFG], [MACS / tot(c) for c in CFG],
              color=["#888", "#888", "tab:blue", "tab:blue", "tab:green", "tab:red"][:len(CFG)])
    ax[1].set_ylabel("MAC / retired instruction")
    ax[1].set_title("work per instruction", fontsize=10)
    for a in ax:
        a.grid(alpha=.3, axis="y")
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_progress.png"), dpi=150)
    plt.close(fig)


def mixfig():
    cls = ["load", "ldpi", "store", "alu", "multiply", "branch", "jump", "qmac8", "qreq"]
    col = dict(load="#4c72b0", ldpi="#55a868", store="#8172b2", alu="#c4c4c4", multiply="#c44e52",
               branch="#dd8452", jump="#ccb974", qmac8="#2ca02c", qreq="#17becf")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = [0.0] * len(CFG)
    for k in cls:
        v = [100.0 * mix(c).get(k, 0) / tot(c) for c in CFG]
        if max(v) < 0.05:
            continue
        ax.bar([NAME[c] for c in CFG], v, bottom=bottom, label=k, color=col[k])
        bottom = [a + b for a, b in zip(bottom, v)]
    ax.set_ylabel("% of retired instructions")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_mix.png"), dpi=150)
    plt.close(fig)


def opsfig():
    groups = ["gemm", "add", "im2col", "maxpool", "pad", "transpose", "avgpool"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bottom = [0.0] * len(CFG)
    for g in groups:
        v = [100.0 * sum(MP[c]["groups"][g].values()) / tot(c) for c in CFG]
        ax.bar([NAME[c] for c in CFG], v, bottom=bottom, label=g)
        bottom = [a + b for a, b in zip(bottom, v)]
    ax.set_ylabel("% of instructions")
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8, loc="lower center")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("share of each operator class (Amdahl)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_ops.png"), dpi=150)
    plt.close(fig)


def cyclesfig():
    import perf_model as PM
    cfgs = [c for c in ("b1", "x0", "x1", "x2", "x2t") if c in MP]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.27
    xs = range(len(cfgs))
    isa = [PM.counts and sum(PM.counts("b1").values()) / sum(PM.counts(c).values()) for c in cfgs]
    cy = [PM.cycles("b1") / PM.cycles(c) for c in cfgs]
    cy40 = [PM.cycles("b1", lat=40) / PM.cycles(c, lat=40) for c in cfgs]
    cy3 = [PM.cycles("b1") / PM.cycles(c, q_mac=3) for c in cfgs]
    ax.bar([x - 1.5 * w / 1 * 0.67 for x in xs], isa, w * .7, label="ISA only (insn ratio)", color="tab:green")
    ax.bar([x - 0.5 * w * 0.67 for x in xs], cy, w * .7, label="model, qmac8=1cy, LAT10", color="tab:blue")
    ax.bar([x + 0.5 * w * 0.67 for x in xs], cy40, w * .7, label="model, qmac8=1cy, LAT40", color="tab:orange")
    ax.bar([x + 1.5 * w * 0.67 for x in xs], cy3, w * .7, label="model, qmac8=3cy, LAT10", color="tab:red")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([NAME[c] for c in cfgs], fontsize=8)
    ax.set_ylabel("speedup vs B1")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3, axis="y")
    ax.set_title("ISA gain vs modelled cycle gain (model, NOT measured)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(F, "fig_cycles.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    progress()
    mixfig()
    opsfig()
    cyclesfig()
    print("figures written to", F)
