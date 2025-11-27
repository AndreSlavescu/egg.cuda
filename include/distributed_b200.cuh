#ifndef DISTRIBUTED_B200_CUH
#define DISTRIBUTED_B200_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_GPUS 8
#define MAX_VAL 127
#define MIN_VAL -127

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
#endif

namespace b200 {

inline void print_topology() {
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    printf("=== B200 Cluster Topology ===\n");
    for (int i = 0; i < num_gpus; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s (SM %.1f, %d SMs, %.1f GB)\n",
            i, prop.name, (float)prop.major + (float)prop.minor / 10.0f,
            prop.multiProcessorCount, (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        
        printf("  NVLink peers: ");
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) continue;
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            if (can_access) printf("%d ", j);
        }
        printf("\n");
    }
    printf("=============================\n\n");
}

inline void enable_p2p_access(int num_gpus) {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }
}

} // namespace b200

__device__ __forceinline__ uint32_t vote_hash_rng(uint32_t seed, uint32_t idx) {
    uint32_t x = seed + idx * 0x9e3779b9;
    x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t vote_gen_noise(uint32_t seed, uint32_t idx) {
    uint32_t r = vote_hash_rng(seed, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__global__ void __launch_bounds__(256)
compute_votes_kernel(int32_t* votes, int out_dim, int in_dim, uint32_t seed, const int* __restrict__ fitnesses, int pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim * in_dim) return;
    
    int o = idx / in_dim;
    int i = idx % in_dim;
    
    int32_t vote = 0;
    for (int p = 0; p < pairs; p++) {
        int f = fitnesses[p];
        if (f == 0) continue;
        uint32_t pair_seed = seed + p;
        int8_t a = vote_gen_noise(pair_seed, o);
        int8_t b = vote_gen_noise(pair_seed, out_dim + i);
        vote += (int32_t)(a * f) * (int32_t)b;
    }
    votes[idx] = vote;
}

__global__ void __launch_bounds__(256)
compute_vector_votes_kernel(int32_t* votes, int len, uint32_t seed, const int* __restrict__ fitnesses, int pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    
    int32_t vote = 0;
    for (int p = 0; p < pairs; p++) {
        int f = fitnesses[p];
        if (f == 0) continue;
        uint32_t pair_seed = seed + p;
        int8_t a = vote_gen_noise(pair_seed, idx);
        vote += (int32_t)a * f;
    }
    votes[idx] = vote;
}

__global__ void __launch_bounds__(256)
accumulate_votes_kernel(int32_t* dst, const int32_t* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void __launch_bounds__(256)
apply_votes_kernel(int8_t* W, const int32_t* votes, int n, int threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t vote = votes[i];
    int8_t w = W[i];
    if (vote > threshold && w < MAX_VAL) W[i] = w + 1;
    else if (vote < -threshold && w > MIN_VAL) W[i] = w - 1;
}

#endif // DISTRIBUTED_B200_CUH
