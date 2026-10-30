#!/usr/bin/env python3
"""Tests of the D-cache simulator ext/cachesweep.cc.

    python3 tests/test_cachesweep.py        (exit 0 = all pass)
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIKE = "/opt/riscv/bin/spike"
GCC = ["riscv64-unknown-elf-gcc", "-march=rv64imc", "-mabi=lp64", "-mcmodel=medany",
       "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2", "-I", "src", "-I", "gen",
       "-T", "src/link.ld", "-Wl,--no-warn-rwx-segments"]
LIBS = ["--extlib=ext/libqmac8.so", "--extlib=ext/libcachesweep.so",
        "--extension=qmac8", "--extension=cachesweep"]
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print("%-4s %-34s got %-26s want %s" % ("ok" if ok else "FAIL", name, got, want))


def sym(elf, name):
    for ln in subprocess.run(["riscv64-unknown-elf-nm", elf], capture_output=True,
                             text=True, check=True).stdout.splitlines():
        a = ln.split()
        if len(a) == 3 and a[2] == name:
            return a[0]


def sweep(elf, cfgs, out, isa="rv64imc", mark=True):
    env = dict(os.environ, CS_CONFIGS=";".join(cfgs), CS_OUT=out)
    if mark:
        env["CS_MARK"] = sym(elf, "phase_mark")
    p = subprocess.run([SPIKE, "--isa=" + isa] + LIBS + [elf], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    r = {}
    for ln in open(os.path.join(ROOT, out)):
        f = ln.split()
        r[f[1]] = tuple(int(v) for v in f[3:8])        # rd wr rdm wrm wb (phase 0)
    return r


def directed():
    for blk, cfgs, want in (
            (0, {"lru": "lru", "fifo": "fifo", "plru": "plru"}, {"lru": 6, "fifo": 5, "plru": 5}),
            (1, {"lru": "lru", "fifo": "fifo"}, {"lru": 4, "fifo": 5})):
        elf = "/tmp/probe%d.elf" % blk
        subprocess.run(GCC + ["-DBLOCK=%d" % blk, "src/startup.S",
                              "src/benchmarks/cache_probe.c", "-o", elf], cwd=ROOT, check=True)
        size, ways = (256, 4) if blk == 0 else (128, 2)
        r = sweep(elf, ["%s,%d,%d,64,%s" % (n, size, ways, p) for n, p in cfgs.items()],
                  "/tmp/probe_cs.txt")
        for n in cfgs:
            check("probe block %d %s misses" % (blk, n), r[n][2] + r[n][3], want[n])


def crosscheck():
    elf = "/tmp/xcheck.elf"
    subprocess.run(GCC + ["-DCFG=3", "-DONLY_LAYER=1", "src/startup.S", "src/layer_kernels.c",
                          "src/benchmarks/run_layers.c", "gen/layers.S", "-o", elf],
                   cwd=ROOT, check=True)
    geoms = ["64:4:64", "64:2:32", "128:8:64", "16:1:128"]
    cfgs = []
    for g in geoms:
        s, w, b = map(int, g.split(":"))
        cfgs.append("g%s,%d,%d,%d,srand" % (g.replace(":", "_"), s * w * b, w, b))
    r = sweep(elf, cfgs, "/tmp/xcheck_cs.txt", mark=False)
    for g in geoms:
        p = subprocess.run([SPIKE, "--isa=rv64imc", "--extlib=ext/libqmac8.so",
                            "--extension=qmac8", "--dc=" + g, elf], cwd=ROOT,
                           capture_output=True, text=True)
        v = {k: int(m) for k, m in re.findall(
            r"D\$ (Read Accesses|Write Accesses|Read Misses|Write Misses|Writebacks):\s+(\d+)",
            p.stdout + p.stderr)}
        want = (v["Read Accesses"], v["Write Accesses"], v["Read Misses"],
                v["Write Misses"], v["Writebacks"])
        check("vs spike --dc=" + g, r["g" + g.replace(":", "_")], want)


if __name__ == "__main__":
    directed()
    crosscheck()
    print("ALL PASSED" if not fails else "%d FAILED" % fails)
    sys.exit(1 if fails else 0)
