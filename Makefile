NVCC ?= $(shell which nvcc 2>/dev/null || \
    ls /usr/local/cuda-13.0/bin/nvcc 2>/dev/null || \
    ls /usr/local/cuda-12.8/bin/nvcc 2>/dev/null || \
    ls /usr/local/cuda/bin/nvcc 2>/dev/null || \
    echo "nvcc")
NVCCFLAGS = -O3 -use_fast_math --expt-relaxed-constexpr
LDFLAGS = -lcublas -lcublasLt

B200_CHECK := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q "10.0" && echo yes)
H100_CHECK := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q "9.0" && echo yes)

ifeq ($(B200_CHECK),yes)
    ARCH = -arch=sm_100
    IS_H100 = 1
else ifeq ($(H100_CHECK),yes)
    ARCH = -arch=sm_90a
    IS_H100 = 1
else
    ARCH = -arch=sm_80
    IS_H100 = 0
endif

ifdef FORCE_ARCH
    ARCH = -arch=$(FORCE_ARCH)
endif

# Verbose PTXAS output
ifdef VERBOSE
    NVCCFLAGS += -Xptxas=-v
endif

# Cutlass kernel naming optimization 
# NVCC has a hotpath for "cutlass" named kernels that targets specific register assignment optimizations
ifdef CUTLASS_OPT
    PREPROCESS = ./scripts/cutlass_rename.sh
else
    PREPROCESS = true
endif

ifdef INT8_TC
    NVCCFLAGS += -DUSE_INT8_TC=1
endif

SRC = full_trained_egg.cu
BUILD_SRC = .build_egg.cu

all: egg_gpu

egg_gpu: $(SRC) int8_tc.cuh
	@echo "Compiling CUDA ($(ARCH))..."
ifdef INT8_TC
	@echo "  INT8 Tensor Core mode enabled"
endif
	@cp $(SRC) $(BUILD_SRC)
	@$(PREPROCESS) $(BUILD_SRC) $(IS_H100)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $(BUILD_SRC)
	@rm -f $(BUILD_SRC)

clean:
	rm -f egg_gpu .build_egg.cu

.PHONY: all clean
