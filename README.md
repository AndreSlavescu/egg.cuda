# EGGROLL in C

A minimalist, dependency-free implementation of the **EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) algorithm family.

**Mission**: To get the most capable pairs of models / platforms cross-implementations of EGGROLL family of algorithms as possible and squeeze every possible bit of performance from the equipment, implementing everything required from the scratch in hardware-optimized fashion.

This project demonstrates **integer-only training** of a language model, completely bypassing the need for standard floating-point arithmetic or heavy ML frameworks like PyTorch or JAX.

## Key Features

*   **Pure C / Bare Metal**: Old-school fashioned and goal-oriented. Zero external dependencies on the CPU side, keeping it close to the metal.
*   **Apple Silicon Optimized**: Vectorized operations using ARM NEON intrinsics and parallelized via Grand Central Dispatch (GCD).
*   **NVIDIA CUDA Optimized**: Custom GPU kernels utilizing Warp-level primitives, Shared Memory, and CUB/Thrust for maximum throughput.
*   **Integer Only**: Operates primarily on `int8` weights/activations with `int32` (CPU) or `int64` (GPU) accumulation—sticking to integer math as long as it yields the best performance for the hardware.
*   **Gradient Free**: Uses Evolution Strategies (ES) with low-rank perturbations instead of backpropagation. It's both wisdom and freedom!

## Quick Start

### 1. Prepare Data
Ensure you have a text dataset named `input.txt` in the current directory.

### 2. Compile & Run

#### Apple Silicon / CPU
```bash
clang -O3 full_trained_egg.c -o egg
./egg
```

#### NVIDIA GPU (CUDA)
```bash
nvcc -O3 full_cuda_train_egg.cu -o egg_cuda
./egg_cuda
```

![Training Output](_imgs_/egg_train.jpeg)

## Configuration

![Configuration](_imgs_/egg_config.jpeg)

## Community & Contributing

We are friendly and welcome all sorts of contributions!

*   **Testers**: Open issues with a description of your available compute, join existing issues if you can platforms described there.
*   **Moderators**: To keep all this under control.
*   **Creatives**: Even if you have nice creative IDEA on README design - you're welcome.

## References

*   **Original JAX Implementation**: [ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg)
*   **Original Paper & Project**: [EGGROLL Website](https://eshyperscale.github.io/)
