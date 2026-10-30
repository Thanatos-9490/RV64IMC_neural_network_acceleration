#!/usr/bin/env python3
"""Per-layer dynamic instruction profile of the kernel, from Spike's PC histogram.

"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKE = "/opt/riscv/bin/spike"
FLAGS = ["--isa=rv64imc", "--extlib=ext/libqmac8.so", "--extension=qmac8"]
GCC = ["riscv64-unknown-elf-gcc", "-march=rv64imc", "-mabi=lp64", "-mcmodel=medany",
       "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2", "-I", "src", "-I", "gen",
       "-T", "src/link.ld", "-Wl,--no-warn-rwx-segments"]
CFG = {"b0": 0, "b1": 1, "x0": 2, "x1": 3, "x2": 5}

LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "c.lw", "c.lwsp", "c.ld", "c.ldsp"}
STORE = {"sb", "sh", "sw", "sd", "c.sw", "c.swsp", "c.sd", "c.sdsp"}
MUL = {"mul", "mulw", "mulh", "mulhsu", "mulhu"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
JUMP = {"jal", "jalr", "c.j", "c.jr", "c.jal", "c.jalr"}


def classify(m):
    if m in LOAD:
        return "load"
    if m in STORE:
        return "store"
    if m in MUL:
        return "multiply"
    if m in BRANCH:
        return "branch"
    if m in JUMP:
        return "jump"
    if m in ("qmac8", "qreq", "ldpi"):
        return m
    return "alu"


def func_ranges(elf, funcs):
    """[(start, end)) of each named function, from the symbol table."""
    out = subprocess.run(["riscv64-unknown-elf-nm", "-S", elf], capture_output=True,
                         text=True, check=True).stdout
    rng = []
    for ln in out.splitlines():
        a = ln.split()
        if len(a) == 4 and a[3] in funcs:
            s = int(a[0], 16)
            rng.append((s, s + int(a[1], 16)))
    return rng


def decode(elf):
    """pc -> mnemonic for every instruction (custom-0 recognised by encoding)."""
    out = subprocess.run(["riscv64-unknown-elf-objdump", "-d", "-M", "no-aliases", elf],
                         capture_output=True, text=True, check=True).stdout
    pcs = {}
    for line in out.splitlines():
        m = re.match(r"^\s+([0-9a-f]+):\s+([0-9a-f]{4,8})\s+(\S+)", line)
        if not m:
            continue
        pc, word, mn = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
        if len(m.group(2)) == 8 and (word & 0x7f) == 0x0b:
            mn = "qmac8" if ((word >> 25) & 0x7f) == 1 else "qreq"
        elif len(m.group(2)) == 8 and (word & 0x7f) == 0x2b:
            mn = "ldpi"
        pcs[pc] = mn
    return pcs


def run_hist(elf):
    p = subprocess.run([SPIKE, "-g"] + FLAGS + [elf], cwd=ROOT, capture_output=True,
                       text=True)
    hist = {}
    for line in p.stderr.splitlines():
        m = re.match(r"^([0-9a-f]+) (\d+)$", line)
        if m:
            hist[int(m.group(1), 16)] = int(m.group(2))
    return hist, p.returncode


def profile(elf, funcs):
    rng = func_ranges(elf, funcs)
    pcs = decode(elf)
    hist, rc = run_hist(elf)
    mix = {}
    for pc, n in hist.items():
        if any(lo <= pc < hi for lo, hi in rng):
            c = classify(pcs.get(pc, "?"))
            mix[c] = mix.get(c, 0) + n
    return mix, rc


def build(cfg, layer):
    elf = "build/p_%s_%d.elf" % (cfg, layer)
    if not os.path.exists(os.path.join(ROOT, elf)):
        subprocess.run(GCC + ["-DCFG=%d" % CFG[cfg], "-DONLY_LAYER=%d" % layer,
                              "src/startup.S", "src/layer_kernels.c",
                              "src/benchmarks/run_layers.c", "gen/layers.S", "-o", elf],
                       cwd=ROOT, check=True, capture_output=True)
    return elf


def main():
    meta = json.load(open(os.path.join(ROOT, "gen/layers.json")))
    funcs = {"run_layer", "gemm_b0", "gemm_b1", "gemm_x0", "gemm_x1", "gemm_x2"}
    res = {}
    for cfg in CFG:
        for i, e in enumerate(meta):
            elf = build(cfg, i)
            mix, rc = profile(os.path.join(ROOT, elf), funcs)
            os.remove(os.path.join(ROOT, elf))
            tot = sum(mix.values())
            res["%s/%s" % (cfg, e["name"])] = dict(total=tot, mix=mix, exit=rc,
                                                  macs=e["macs"], outputs=e["M"] * e["N"])
            print("%-3s %-9s total=%-12d exit=%d" % (cfg, e["name"], tot, rc), flush=True)
    json.dump(res, open(os.path.join(ROOT, "results/processed/layers_profile.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
