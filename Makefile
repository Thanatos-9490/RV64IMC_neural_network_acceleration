# RV64IMC Quantized-NN Acceleration Project

RISCV_GCC := riscv64-unknown-elf-gcc
SPIKE     := /opt/riscv/bin/spike
HOST_CC   := gcc
RISCV_FLAGS   := -march=rv64imc -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -ffreestanding -O2
RISCV_LDFLAGS := -T src/link.ld -Wl,--no-warn-rwx-segments
SPIKE_FLAGS   := --isa=rv64imc --extlib=ext/libqmac8.so --extension=qmac8
INC           := -I src -I gen
CFGNAME       := b0 b1 x0 x1
CFGNUM_b0 := 0
CFGNUM_b1 := 1
CFGNUM_x0 := 2
CFGNUM_x1 := 3
CFGNUM_x1t := 4
CFGNUM_x2 := 5
CFGNUM_x2t := 6

.DEFAULT_GOAL := all
.PHONY: spike_verify_gemm spike_verify_ldpi all extlib cachelib host_test spike_verify layers model_infer profile micro model quantize export clean test_cache cache model_profile
SHELL := /bin/bash
all: extlib host_test spike_verify spike_verify_ldpi spike_verify_gemm layers model_infer

ext/libqmac8.so: ext/qmac8.cc src/tfl_requant.h
	@g++ -std=c++2a -fPIC -shared -O2 -I src -o $@ $< -I/opt/riscv/riscv-isa-sim/riscv -I/opt/riscv/riscv-isa-sim/build
extlib: ext/libqmac8.so
ext/libcachesweep.so: ext/cachesweep.cc
	@g++ -std=c++2a -fPIC -shared -O2 -o $@ $< -I/opt/riscv/riscv-isa-sim/riscv -I/opt/riscv/riscv-isa-sim/build
cachelib: ext/libcachesweep.so

build/host_%.o: src/%.c src/nnaccel.h src/tfl_requant.h src/kernels.h
	@mkdir -p build
	@$(HOST_CC) -O2 -c $< -o $@ $(INC)
build/host_test_kernels_host.o: tests/test_kernels_host.c src/kernels.h src/nnaccel.h gen/qreq_vectors.h
	@mkdir -p build
	@$(HOST_CC) -O2 -c $< -o $@ $(INC)
build/test_kernels_host: build/host_kernels.o build/host_qmac8_model.o build/host_test_kernels_host.o
	@$(HOST_CC) $^ -o $@
host_test: build/test_kernels_host
	@echo "=== host: software model vs reference ==="
	@./build/test_kernels_host

build/startup.o: src/startup.S
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) -c $< -o $@
build/rv_%.o: src/%.c src/nnaccel.h src/tfl_requant.h src/kernels.h src/layer_kernels.h
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) $(INC) -c $< -o $@
build/layers_data.o: gen/layers.S $(wildcard gen/*.bin)
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) -c $< -o $@
BM := build/startup.o build/rv_kernels.o

build/verify_qmac8.o: src/benchmarks/verify_qmac8.c src/nnaccel.h src/tfl_requant.h gen/qreq_vectors.h
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) $(INC) -c $< -o $@
build/verify_qmac8.elf: $(BM) build/verify_qmac8.o
	@$(RISCV_GCC) $(RISCV_FLAGS) $(RISCV_LDFLAGS) $^ -o $@
spike_verify: ext/libqmac8.so build/verify_qmac8.elf
	@echo "=== Spike: qmac8 / qreq vs reference ==="
	@$(SPIKE) $(SPIKE_FLAGS) build/verify_qmac8.elf && echo "--- PASSED ---"

# ---- X2 instruction ldpi: directed + random + trap tests ----------------------
build/verify_ldpi.elf: src/benchmarks/verify_ldpi.c src/startup.S src/link.ld
	@mkdir -p build
	@$(RISCV_GCC) -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -ffreestanding -O1 $(RISCV_LDFLAGS) src/startup.S src/benchmarks/verify_ldpi.c -o $@
spike_verify_ldpi: ext/libqmac8.so build/verify_ldpi.elf
	@echo "=== Spike: ldpi (X2) directed / random / precise-trap tests ==="
	@$(SPIKE) --isa=rv64imc_zicsr --extlib=ext/libqmac8.so --extension=qmac8 build/verify_ldpi.elf && echo "--- PASSED ---"

# ---- randomized GEMM-level verification of all kernel configs ---------------
build/verify_gemm_random.elf: src/benchmarks/verify_gemm_random.c src/layer_kernels.c src/layer_kernels.h src/nnaccel.h src/tfl_requant.h src/startup.S
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) $(INC) $(RISCV_LDFLAGS) src/startup.S src/layer_kernels.c src/benchmarks/verify_gemm_random.c -o $@
spike_verify_gemm: ext/libqmac8.so build/verify_gemm_random.elf
	@echo "=== Spike: randomized GEMM (all 7 configs) vs plain-C reference ==="
	@$(SPIKE) $(SPIKE_FLAGS) build/verify_gemm_random.elf && echo "--- PASSED ---"

# CFG: 0=B0 1=B1 2=X0 3=X1   (ONLY_LAYER=i for single-layer ELFs used by profiling)
build/layers_%.elf: src/benchmarks/run_layers.c src/layer_kernels.c src/layer_kernels.h src/nnaccel.h src/tfl_requant.h src/startup.S gen/layers.S gen/layers.h $(wildcard gen/*.bin)
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) $(INC) -DCFG=$(CFGNUM_$*) $(RISCV_LDFLAGS) src/startup.S src/layer_kernels.c src/benchmarks/run_layers.c gen/layers.S -o $@
LAYER_CFGS := $(CFGNAME) x2
layers: ext/libqmac8.so $(addprefix build/layers_,$(addsuffix .elf,$(LAYER_CFGS)))
	@echo "=== ResNet-18 W8A8 layers vs TFLite ==="
	@for c in $(LAYER_CFGS); do $(SPIKE) $(SPIKE_FLAGS) build/layers_$$c.elf; r=$$?; echo "  config $$c : exit=$$r"; [ $$r -eq 0 ] || exit 1; done
	@echo "--- BIT-EXACT (all configs) ---"

# ---- full-network inference (all 37 ops chained) -----------------------------
# Needs Zicntr (instret) in Spike; the GEMM kernels are the same code as `layers`.
# B0/B1/X0 run 1 image (per-op check, ~minutes); X1 runs all 10 images.
MODEL_SRC := src/startup.S src/layer_kernels.c src/model_ops.c src/benchmarks/run_model.c gen/model.S
MODEL_FLAGS := -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany -nostdlib -nostartfiles -ffreestanding -O2
SPIKE_MODEL := --isa=rv64imc_zicntr --extlib=ext/libqmac8.so --extension=qmac8
MODEL_NIMG_b0 := 1
MODEL_NIMG_b1 := 1
MODEL_NIMG_x0 := 1
MODEL_NIMG_x1 := 10
MODEL_NIMG_x1t := 10
MODEL_NIMG_x2 := 10
MODEL_NIMG_x2t := 10
build/model_%.elf: $(MODEL_SRC) src/layer_kernels.h src/model_ops.h src/htif.h src/nnaccel.h src/tfl_requant.h gen/model.h $(wildcard gen/*.bin)
	@mkdir -p build
	@$(RISCV_GCC) $(MODEL_FLAGS) $(INC) -DCFG=$(CFGNUM_$*) -DNIMG=$(MODEL_NIMG_$*) $(RISCV_LDFLAGS) $(MODEL_SRC) -o $@
MODEL_CFGS := $(CFGNAME) x1t x2 x2t
model_infer: ext/libqmac8.so $(addprefix build/model_,$(addsuffix .elf,$(MODEL_CFGS)))
	@echo "=== ResNet-18 W8A8: FULL network, chained, vs TFLite (per-op + logits) ==="
	@for c in $(MODEL_CFGS); do echo "--- config $$c"; $(SPIKE) $(SPIKE_MODEL) build/model_$$c.elf | grep -v '^IMAGE'; r=$${PIPESTATUS[0]}; echo "  config $$c : exit=$$r"; [ $$r -eq 0 ] || exit 1; done
	@echo "--- FULL MODEL BIT-EXACT (all configs) ---"

profile: ext/libqmac8.so
	python3 scripts/profile_pc.py

build/micro_%.o: src/benchmarks/bench_dot.c src/kernels.h src/nnaccel.h
	@mkdir -p build
	@$(RISCV_GCC) $(RISCV_FLAGS) $(INC) -DUSE_QMAC8=$(if $(filter b0,$*),0,1) -c $< -o $@
build/micro_%.elf: $(BM) build/micro_%.o
	@$(RISCV_GCC) $(RISCV_FLAGS) $(RISCV_LDFLAGS) $^ -o $@
micro: ext/libqmac8.so build/micro_b0.elf build/micro_x1.elf
	python3 scripts/profile_stream.py build/micro_b0.elf build/micro_x1.elf

model:
	python scripts/model_info.py models/resnet18.tflite --csv results/processed/resnet18_layers.csv > results/processed/resnet18_analysis.txt
quantize:
	python scripts/quantize_model.py resnet18
export:
	python scripts/export_layers.py
# ---- cache design exploration (needs matplotlib in python3 for the figures) --
test_cache: ext/libqmac8.so ext/libcachesweep.so
	python3 tests/test_cachesweep.py
cache: ext/libqmac8.so ext/libcachesweep.so
	python3 scripts/cache_sweep.py --cfgs x1,x1t,b1     # ~10 min
	python3 scripts/profile_model.py                    # ~6 min, needed for CPI model
	python3 scripts/cache_summary.py
model_profile: ext/libqmac8.so
	python3 scripts/profile_model.py
	python3 scripts/summarize_model.py

clean:
	rm -rf build ext/*.so results/raw/*.log results/raw/*.txt


asm: asm/kernels.rv64imc.txt asm/layers.b0.txt asm/layers.x1.txt
	@echo "Assembly dumps generated in asm/"

asm/kernels.rv64imc.txt: src/layer_kernels.c src/layer_kernels.h src/nnaccel.h src/tfl_requant.h
	python3 scripts/dump_asm.py --kernels

asm/layers.b0.txt: build/layers_b0.elf
	python3 scripts/dump_asm.py --elf b0

asm/layers.x1.txt: build/layers_x1.elf
	python3 scripts/dump_asm.py --elf x1

clean_asm:
	rm -rf asm/

dump:
	@echo "Disassembling $(TARGET) to dump.txt …"
	objdump -d $(BUILD_DIR)/$(TARGET) > dump.txt
