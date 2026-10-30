#!/usr/bin/env python3
"""D-cache design-space exploration on the chained ResNet-18 W8A8 inference.

One Spike run of build/model_<cfg>.elf feeds every data load/store to all cache
models at once (ext/cachesweep.cc).  Accesses are attributed to the graph op
(phase = op index) through 1-byte stores to `phase_mark`.

    python3 scripts/cache_sweep.py                  -> results/processed/cache_sweep.json
    python3 scripts/cache_sweep.py --smoke          -> a few configs, ~1 min

Sweeps (all write-back / write-allocate, accesses of image 0 only):
  size       2..64 KB   @ 4-way, 64 B line, PLRU       (CV-Wally default = 16 KB)
  assoc      1..16 way  @ 16 KB, 64 B line, PLRU
  line       16..128 B  @ 16 KB, 4-way, PLRU
  policy     LRU / PLRU / FIFO / RANDOM  @ 4 KB .. 32 KB, 4-way, 64 B   (small
             caches expose the policy; at 16 KB the weights stream anyway)
and the same sweep for B1 (scalar) and X1T (cache-blocked) to show how the
sensitivity depends on the software loop order.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKE = "/opt/riscv/bin/spike"
KB = 1024


def configs(smoke):
    """name -> (size, ways, line, policy); name encodes the geometry."""
    c = {}

    def add(size, ways, line, pol):
        c["%dK-%dw-%dB-%s" % (size // KB, ways, line, pol)] = (size, ways, line, pol)

    if smoke:
        for s in (4, 16):
            add(s * KB, 4, 64, "plru")
        add(16 * KB, 4, 64, "lru")
        return c
    for s in (2, 4, 8, 16, 32, 64):                       # size
        add(s * KB, 4, 64, "plru")
    for w in (1, 2, 4, 8, 16):                            # associativity
        add(16 * KB, w, 64, "plru")
    for ln in (16, 32, 64, 128):                          # line size
        add(16 * KB, 4, ln, "plru")
    for s in (4, 8, 16, 32):                              # replacement policy
        for pol in ("lru", "plru", "fifo", "rand"):
            add(s * KB, 4, 64, pol)
    for w in (2, 8):                                      # policy x associativity
        for pol in ("lru", "plru", "fifo", "rand"):
            add(16 * KB, w, 64, pol)
    return c


def symbol(elf, name):
    out = subprocess.run(["riscv64-unknown-elf-nm", elf], capture_output=True,
                         text=True, check=True).stdout
    for ln in out.splitlines():
        a = ln.split()
        if len(a) == 3 and a[2] == name:
            return a[0]
    raise SystemExit("symbol %s not found in %s" % (name, elf))


def run(elf, cfgs, out):
    spec = ";".join("%s,%d,%d,%d,%s" % ((n,) + v) for n, v in cfgs.items())
    env = dict(os.environ, CS_CONFIGS=spec, CS_MARK=symbol(elf, "phase_mark"),
               CS_OUT=out)
    t = time.time()
    p = subprocess.run([SPIKE, "--isa=rv64imc_zicntr", "--extlib=ext/libqmac8.so",
                        "--extlib=ext/libcachesweep.so", "--extension=qmac8",
                        "--extension=cachesweep", elf], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if p.returncode:
        raise SystemExit("spike failed (%d): %s" % (p.returncode, p.stdout[-300:]))
    res = {}
    for ln in open(os.path.join(ROOT, out)):
        f = ln.split()
        if f[0] != "CS":
            continue
        n, ph = f[1], int(f[2])
        rd, wr, rdm, wrm, wb = map(int, f[3:8])
        res.setdefault(n, {})[ph] = [rd, wr, rdm, wrm, wb]
    print("  %-4s %d caches, %.0fs" % (os.path.basename(elf), len(cfgs), time.time() - t),
          flush=True)
    return res


def build(cfg):
    """ELF running exactly ONE image (phases = ops of image 0, cold cache at start)."""
    num = {"b0": 0, "b1": 1, "x0": 2, "x1": 3, "x1t": 4, "x2": 5, "x2t": 6}[cfg]
    elf = "build/cs_%s.elf" % cfg
    gcc = ["riscv64-unknown-elf-gcc", "-march=rv64imc_zicsr", "-mabi=lp64",
           "-mcmodel=medany", "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2",
           "-I", "src", "-I", "gen", "-DCFG=%d" % num, "-DNIMG=1", "-T", "src/link.ld",
           "-Wl,--no-warn-rwx-segments", "src/startup.S", "src/layer_kernels.c",
           "src/model_ops.c", "src/benchmarks/run_model.c", "gen/model.S", "-o", elf]
    subprocess.run(gcc, cwd=ROOT, check=True)
    return elf, num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cfgs", default="x1,x1t,b1")
    a = ap.parse_args()
    cfgs = configs(a.smoke)
    dst = "cache_sweep_smoke.json" if a.smoke else "cache_sweep.json"
    path = os.path.join(ROOT, "results/processed", dst)
    out = {"configs": {n: list(v) for n, v in cfgs.items()}, "runs": {}}
    if os.path.exists(path):                  # merge: re-run only some workloads
        old = json.load(open(path))
        if old["configs"] == out["configs"]:
            out["runs"] = old["runs"]
    os.makedirs(os.path.join(ROOT, "results/raw"), exist_ok=True)
    for cfg in a.cfgs.split(","):
        elf, _ = build(cfg)
        out["runs"][cfg] = run(elf, cfgs, "results/raw/cs_%s.txt" % cfg)
    json.dump(out, open(path, "w"))
    print("wrote results/processed/" + dst)


if __name__ == "__main__":
    main()
