#!/usr/bin/env python3
"""Independent cross-check of the PC-histogram instruction counts.

"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = {"b0": 0, "b1": 1, "x0": 2, "x1": 3, "x1t": 4, "x2": 5, "x2t": 6}
GCC = ["riscv64-unknown-elf-gcc", "-march=rv64imc_zicsr", "-mabi=lp64", "-mcmodel=medany",
       "-nostdlib", "-nostartfiles", "-ffreestanding", "-O2", "-I", "src", "-I", "gen",
       "-T", "src/link.ld", "-Wl,--no-warn-rwx-segments", "src/startup.S",
       "src/layer_kernels.c", "src/model_ops.c", "src/benchmarks/run_model.c",
       "gen/model.S"]
SPIKE = ["/opt/riscv/bin/spike", "--isa=rv64imc_zicntr", "--extlib=ext/libqmac8.so",
         "--extension=qmac8"]


def main():
    prof = json.load(open(os.path.join(ROOT, "results/processed/model_profile.json")))
    res, bad = {}, 0
    for cfg, n in NUM.items():
        elf = "build/ir_%s.elf" % cfg
        subprocess.run(GCC + ["-DCFG=%d" % n, "-DNIMG=1", "-DREPORT_INSTRET", "-o", elf],
                       cwd=ROOT, check=True)
        p = subprocess.run(SPIKE + [elf], cwd=ROOT, capture_output=True, text=True)
        os.remove(os.path.join(ROOT, elf))
        line = [l for l in p.stdout.splitlines() if l.startswith("INSTRET")]
        got = int(line[0].split()[1])
        want = sum(sum(m.values()) for m in prof[cfg]["groups"].values()) if cfg in prof else None
        raw = None if want is None else got - want
        # The read pair costs a constant 3 retired instructions per op that the
        # histogram (which counts exec_op only) does not see: the second rdinstret,
        # the reload of the first value and the subtract (see the objdump of main).
        # run_model.c already removes the 'back-to-back read' part; the remaining
        # compiler-inserted ld/add is a property of this build and is reported, not hidden.
        diff = None if raw is None else raw - 37 * 3
        res[cfg] = dict(instret=got, histogram=want, raw_diff=raw, diff_after_37x3=diff)
        bad += diff not in (None, 0)
        print("%-4s instret=%-12d histogram=%-12s raw diff=%s (37 ops x 3 = 111) residual=%s" % (cfg, got, want, raw, diff), flush=True)
    json.dump(res, open(os.path.join(ROOT, "results/processed/instret_check.json"), "w"),
              indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
