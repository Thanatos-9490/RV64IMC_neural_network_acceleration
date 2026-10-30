#!/usr/bin/env python3
"""Streaming instruction profile of an ELF run under Spike (-l log, not stored).

"""
import json
import re
import subprocess
import sys

ROOT = __file__.rsplit("/scripts/", 1)[0]
SPIKE = "/opt/riscv/bin/spike"
FLAGS = ["-l", "--isa=rv64imc", "--extlib=ext/libqmac8.so", "--extension=qmac8"]

LOAD = {"lb", "lbu", "lh", "lhu", "lw", "lwu", "ld", "c.lw", "c.lwsp", "c.ld", "c.ldsp"}
STORE = {"sb", "sh", "sw", "sd", "c.sw", "c.swsp", "c.sd", "c.sdsp"}
MUL = {"mul", "mulw", "mulh", "mulhsu", "mulhu"}
BRANCH = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
JUMP = {"jal", "jalr", "c.j", "c.jr", "c.jal", "c.jalr"}
INSN = re.compile(rb"^core\s+\d+:\s+0x[0-9a-f]+\s+\(0x[0-9a-f]+\)\s+(\S+)")
EXEC = re.compile(rb"Executed\s+(\d+)\s+times")


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
    if m in ("qmac8", "qreq"):
        return m
    return "alu"


def profile(elf):
    p = subprocess.Popen([SPIKE] + FLAGS + [elf], cwd=ROOT, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE, bufsize=1 << 20)
    mix, inside, last = {}, False, None
    for line in p.stderr:
        if b">>>>" in line:
            if b"b_begin" in line:
                inside = True
            elif b"b_end" in line:
                inside = False
            continue
        m = INSN.match(line)
        if m:
            last = classify(m.group(1).decode())
            if inside:
                mix[last] = mix.get(last, 0) + 1
            continue
        m = EXEC.search(line)
        if m and inside and last is not None:
            mix[last] = mix.get(last, 0) + int(m.group(1)) - 1   # first one already counted
    p.wait()
    return mix, p.returncode


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        mix, rc = profile("build/b0_dot.elf")
        t = sum(mix.values())
        print("stream total =", t, "| stored-log total = 462001 ->", "MATCH" if t == 462001 else "MISMATCH")
        sys.exit(0 if t == 462001 else 1)
    out = {}
    for elf in sys.argv[1:]:
        mix, rc = profile(elf)
        out[elf] = dict(total=sum(mix.values()), mix=mix, exit=rc)
        print(elf, sum(mix.values()), mix, "exit", rc, flush=True)
    json.dump(out, open(ROOT + "/results/processed/micro_profile.json", "w"), indent=1)
