# RV64IMC_neural_network_acceleration

ISA extension + Spike implementation + instruction-level evaluation for
**ResNet-18, W8A8**.  See `docs/DESIGN.md` (decisions, encodings) and
`docs/EXPERIMENTS.md` (measurements and their limits).


## Cache design exploration (extension, +5%)

`make cache` sweeps size / associativity / line size / LRU-PLRU-FIFO-RANDOM on the
full inference (31 caches per run, B1 vs X1 vs X1T).  Main finding: the same
workload is **cache-insensitive in B1 (1.0% miss at 16 KB) but cache-bound after
qmac8** (7.5% miss, 6.5x higher MPKI) because the weight matrix of most layers
(36 KB-2.3 MB) is re-streamed once per output row; a cache-blocked loop order (X1T,
same results) cuts the miss rate 2.4x (7.5% -> 3.2%).  Under a stated miss-penalty model the
instruction-count gain B1->X1 of 6.45x becomes 3.5x-4.8x.  Tables:
`results/processed/cache_summary.md`, figures: `results/figures/`.

Details and caveats: `docs/EXPERIMENTS.md`.

## Run

```bash
conda activate aca_term_prj      # only needed for model / quantize / export
make                 # extlib + host_test + spike_verify + layers (bit-exact check)
make model_infer     # full network, 7 configs, per-op + logits check (~1 min, part of `make`)
make profile         # per-layer x 5 configs instruction profile (~5 min)
python3 scripts/summarize.py     # -> results/processed/summary.md
make model_profile   # whole-model profile per operator class (~8 min) -> model_summary.md
make test_cache      # cache simulator: hand-computed probes + equality with `spike --dc`
make cache           # cache sweep (~10 min) -> cache_summary.md + figures (needs matplotlib)
make micro           # dot-product microbenchmark
make asm             # dump assembly of kernel functions and whole ELFs (asm/)
python scripts/export_model.py   # (conda) re-export the whole graph into gen/model.*
```

`make` needs only the RISC-V toolchain and Spike (`/opt/riscv`); the exported
layers and the whole-graph export are committed in `gen/`.  The figures need
`matplotlib` in the system `python3` (it is not in the conda env).

## Deliverables map

| Brief item | Where |
|---|---|
| Report (6-8 page style) | `report/report.md` (+ `.html`, `.docx`) |
| Design / ISA spec | `docs/DESIGN.md` |
| Experiments, caveats | `docs/EXPERIMENTS.md`, `experiments/README.md` |
| AI usage | `AI_USAGE.md` |
| Results, figures | `results/processed/`, `results/figures/` |

## Layout

```
ext/qmac8.cc            Spike extension: qmac8, qreq (+ disassembly)
src/tfl_requant.h       single canonical TFLite requantization (C)
src/nnaccel.h           inline .insn wrappers, host model switch
src/layer_kernels.c     B0 / B1 / X0 / X1 GEMM kernels
ext/cachesweep.cc       Spike extension: multi-config D-cache simulator (per-op counters)
src/kernels.c,h         dot-product kernels (microbenchmark, host test)
src/model_ops.c,h       transpose / pad / im2col / maxpool / add / avgpool (TFLite-exact)
src/htif.h              console output + instret for bare-metal Spike programs
src/benchmarks/         verify_qmac8.c, run_layers.c, run_model.c, cache_probe.c, bench_dot.c
tests/                  host-side test of the software model
scripts/quantize_model.py   PTQ + accuracy        (TensorFlow)
scripts/export_layers.py    .tflite -> gen/ blobs (TensorFlow)
scripts/export_model.py     whole graph -> gen/model.{h,S} + golden checksums (TensorFlow)
scripts/profile_model.py    whole-model profile per operator; summarize_model.py
scripts/cache_sweep.py      runs the cache sweep; cache_summary.py tables + figures
scripts/tflq.py             independent numpy requant reference
scripts/profile_pc.py       instruction profile from Spike PC histogram
scripts/summarize.py        results/processed/summary.md
gen/                    21 exported ResNet-18 layers + qreq golden vectors
models/                 resnet18{,_w8a8,_w8a16}.tflite, mobilenet_v3_large.tflite
```

