NVCC ?= $(shell which nvcc 2>/dev/null || \
    ([ -x /usr/local/cuda-13.0/bin/nvcc ] && echo /usr/local/cuda-13.0/bin/nvcc) || \
    ([ -x /usr/local/cuda-12.8/bin/nvcc ] && echo /usr/local/cuda-12.8/bin/nvcc) || \
    echo /usr/local/cuda/bin/nvcc)

NVCCFLAGS = -O3 -use_fast_math --expt-relaxed-constexpr -Iinclude
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

ifdef VERBOSE
    NVCCFLAGS += -Xptxas=-v
endif

ifdef CUTLASS_OPT
    PREPROCESS = ./scripts/cutlass_rename.sh
else
    PREPROCESS = true
endif

ifdef INT8_TC
    NVCCFLAGS += -DUSE_INT8_TC=1
endif

ifdef USE_DISTRIBUTED
    NVCCFLAGS += -DUSE_DISTRIBUTED=1 -Xcompiler -fopenmp
    LDFLAGS += -lnccl -lgomp
endif

ifdef B200_OPTIMIZATIONS
    NVCCFLAGS += -DB200_OPTIMIZATIONS=1
endif

SRC_DIR = csrc
INCLUDE_DIR = include
SRC = $(SRC_DIR)/full_trained_egg.cu
BUILD_SRC = .build_egg.cu

HEADERS = $(INCLUDE_DIR)/int8_tc.cuh $(INCLUDE_DIR)/distributed_b200.cuh

all: egg_gpu

egg_gpu: $(SRC) $(HEADERS)
	@echo "Compiling CUDA ($(ARCH))..."
ifdef INT8_TC
	@echo "  INT8 Training enabled"
endif
ifdef USE_DISTRIBUTED
	@echo "  Distributed multi-GPU mode enabled"
endif
ifdef B200_OPTIMIZATIONS
	@echo "  B200 optimizations enabled"
endif
	@cp $(SRC) $(BUILD_SRC)
	@$(PREPROCESS) $(BUILD_SRC) $(IS_H100)
	$(NVCC) $(NVCCFLAGS) $(ARCH) $(LDFLAGS) -o $@ $(BUILD_SRC)
	@rm -f $(BUILD_SRC)

clean:
	rm -f egg_gpu .build_egg.cu

.PHONY: all clean
