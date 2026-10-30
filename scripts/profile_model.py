#!/usr/bin/env python3
"""Whole-network dynamic-instruction profile (all 37 ops chained), per operator class.

"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_pc as P                                          # noqa: E402

ROOT = P.ROOT
FLAGS = ["--isa=rv64imc_zicntr", "--extlib=ext/libqmac8.so", "--extension=qmac8"]
GCC = ["riscv64-unknown-elf-gcc", "-march=rv64imc_zicsr", "-mabi=lp64",
       "-mcmodel=medany", "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2",
       "-I", "src", "-I", "gen", "-T", "src/link.ld", "-Wl,--no-warn-rwx-segments"]
SRC = ["src/startup.S", "src/layer_kernels.c", "src/model_ops.c",
       "src/benchmarks/run_model.c", "gen/model.S"]
GEMM = {"run_layer", "gemm_b0", "gemm_b1", "gemm_x0", "gemm_x1", "gemm_x1t", "gemm_x2", "gemm_x2t"}
GROUPS = {"gemm": GEMM, "im2col": {"op_im2col"}, "pad": {"op_pad"},
          "transpose": {"op_transpose"}, "maxpool": {"op_maxpool"},
          "add": {"op_add"}, "avgpool": {"op_avgpool"}, "dispatch": {"exec_op"}}
VERIFY = {"check_op", "main"}


def build(cfg, extra=()):
    elf = "build/pm_%s.elf" % cfg
    subprocess.run(GCC + ["-DCFG=%d" % P.CFG[cfg], "-DNIMG=1"] + list(extra) + SRC +
                   ["-o", elf], cwd=ROOT, check=True, capture_output=True)
    return elf


def fn_table(elf):
    out = subprocess.run(["riscv64-unknown-elf-nm", "-S", elf], capture_output=True,
                         text=True, check=True).stdout
    tab = []
    for ln in out.splitlines():
        a = ln.split()
        if len(a) == 4 and a[2].lower() in ("t",):
            # GCC clones (exec_op.constprop.0, gemm_x1.isra.0, ...) -> base name
            tab.append((int(a[0], 16), int(a[0], 16) + int(a[1], 16), a[3].split(".")[0]))
    return tab


def profile(cfg, extra=()):
    elf = os.path.join(ROOT, build(cfg, extra))
    tab = fn_table(elf)
    pcs = P.decode(elf)
    p = subprocess.run([P.SPIKE, "-g"] + FLAGS + [elf], cwd=ROOT, capture_output=True,
                       text=True)
    hist = {}
    for line in p.stderr.splitlines():
        m = __import__("re").match(r"^([0-9a-f]+) (\d+)$", line)
        if m:
            hist[int(m.group(1), 16)] = int(m.group(2))
    byfn = {}
    for pc, n in hist.items():
        name = next((f for lo, hi, f in tab if lo <= pc < hi), "?")
        cls = P.classify(pcs.get(pc, "?"))
        d = byfn.setdefault(name, {})
        d[cls] = d.get(cls, 0) + n
    os.remove(elf)
    return byfn, p.returncode


def main():
    res = {}
    P.CFG["x1t"] = 4
    P.CFG["x2t"] = 6
    for cfg in P.CFG:
        byfn, rc = profile(cfg)
        grp = {}
        for g, fns in GROUPS.items():
            mix = {}
            for f in fns:
                for c, n in byfn.get(f, {}).items():
                    mix[c] = mix.get(c, 0) + n
            grp[g] = mix
        verify = sum(sum(byfn.get(f, {}).values()) for f in VERIFY)
        other = {f: sum(v.values()) for f, v in byfn.items()
                 if f not in VERIFY and not any(f in s for s in GROUPS.values())}
        res[cfg] = dict(exit=rc, groups=grp, verify_insns=verify, other=other)
        tot = sum(sum(m.values()) for m in grp.values())
        print("%-3s exit=%d  inference=%d  verify(excluded)=%d  other=%s"
              % (cfg, rc, tot, verify, other), flush=True)
    json.dump(res, open(os.path.join(ROOT, "results/processed/model_profile.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
