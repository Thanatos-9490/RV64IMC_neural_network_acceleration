#!/usr/bin/env python3
"""
Dump assembly of the kernel functions for each config.
Usage: python3 scripts/dump_asm.py [--kernels] [--elf config]
"""
import subprocess, os, sys, tempfile

RISCV_GCC = "riscv64-unknown-elf-gcc"
RISCV_FLAGS = "-march=rv64imc -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -ffreestanding -O2"
INC = "-I src -I gen"
SOURCE = "src/layer_kernels.c"

def compile_config(cfg_id):
    """Return objdump text of the whole .text section."""
    with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as tmp:
        tmp.close()
        cmd = f"{RISCV_GCC} {RISCV_FLAGS} {INC} -DCFG_ONLY={cfg_id} -c {SOURCE} -o {tmp.name}"
        subprocess.check_call(cmd, shell=True)
        out = subprocess.check_output(
            f"riscv64-unknown-elf-objdump -d -M no-aliases {tmp.name}",
            shell=True, text=True)
        os.unlink(tmp.name)
        return out

def extract_gemm_section(full_text):
    """Find the part from <run_layer> to end of .text."""
    lines = full_text.split('\n')
    start = None
    for i, line in enumerate(lines):
        if '<run_layer>:' in line:
            start = i
            break
    if start is None:
        return full_text
    # Continue until a line that looks like another section label,
    # or blank line followed by a new function label.
    out_lines = []
    in_text = True
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == '':
            # keep blank line
            out_lines.append(line)
            continue
        if stripped.startswith('Disassembly of section') and 'text' not in stripped:
            break
        # Another function label not containing 'gemm'? we still include it,
        # but break when we see a new section.
        if ':' in stripped and stripped.endswith(':'):
            # This is a label; if it's not run_layer and not gemm_*, it's probably
            # a static helper (like .Lxxx). Keep.
            pass
        out_lines.append(line)
    return '\n'.join(out_lines)

def dump_elf_config(name):
    """Dump full ELF disassembly."""
    elf = f"build/layers_{name}.elf"
    if not os.path.exists(elf):
        print(f"ELF {elf} not found, skipping")
        return
    out = subprocess.check_output(
        f"riscv64-unknown-elf-objdump -d -M no-aliases {elf}",
        shell=True, text=True)
    with open(f"asm/layers.{name}.txt", 'w') as f:
        f.write(out)
    print(f"Generated asm/layers.{name}.txt")

def main():
    os.makedirs("asm", exist_ok=True)
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        return
    if '--kernels' in sys.argv or len(sys.argv) == 1:
        cfg_map = {'b0':0, 'b1':1, 'x0':2, 'x1':3}
        for name, cfg in cfg_map.items():
            print(f"Compiling config {name} (CFG={cfg})...")
            full = compile_config(cfg)
            extracted = extract_gemm_section(full)
            with open(f"asm/kernel_{name}.txt", 'w') as f:
                f.write(extracted)
            # Also write combined
        with open("asm/kernels.rv64imc.txt", 'w') as f:
            f.write("=== Assembly of config B0, B1, X0, X1 (run_layer + kernel functions) ===\n")
            for name in cfg_map.keys():
                f.write(f"\n--- {name.upper()} ---\n")
                with open(f"asm/kernel_{name}.txt") as inf:
                    f.write(inf.read())
        print("Generated asm/kernels.rv64imc.txt")
    if '--elf' in sys.argv:
        for name in ['b0','b1','x0','x1']:
            dump_elf_config(name)
    else:
        for name in ['b0','x1']:
            dump_elf_config(name)

if __name__ == '__main__':
    main()