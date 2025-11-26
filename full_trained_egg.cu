/**
 * CUDA EGGROLL - Optimized with Parallel RNG + H100 TMA support
 * 
 * Key optimizations:
 * 1. Hash-based RNG for O(1) noise access
 * 2. Vectorized loads (float4) for memory bandwidth
 * 3. TMA async copies on H100 (SM90+)
 * 4. cuBLAS TF32 for matrix multiplies
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Optional INT8 Tensor Core support (H100 only)
#ifdef USE_INT8_TC
#include "int8_tc.cuh"
#endif

// Check for H100 (SM90) at compile time
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define USE_TMA 1
#else
#define USE_TMA 0
#endif

#define VOCAB_SIZE 256
#define HIDDEN_DIM 256
#define N_LAYERS 6
#define SEQ_LEN 100
#define POPULATION_SIZE 1024
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define UPDATE_THRESHOLD 160
#define MAX_VAL 127
#define MIN_VAL -127
#define MLP_EXPAND_DIM (HIDDEN_DIM * 4)
#define PAIRS (POPULATION_SIZE / 2)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

__constant__ int32_t d_EXP2_TABLE[256];

__device__ __forceinline__ float clip_f(float v) {
    return fminf(fmaxf(v, -127.0f), 127.0f);
}

// Hash-based RNG - O(1) access to any position
__device__ __forceinline__ uint32_t hash_rng(uint32_t seed, uint32_t idx) {
    uint32_t x = seed + idx * 0x9e3779b9;
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t gen_noise_hash(uint32_t seed, uint32_t idx) {
    uint32_t r = hash_rng(seed, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

// Original xorshift for compatibility with weight updates
__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

__device__ __forceinline__ int8_t gen_noise(uint32_t r) {
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__device__ __forceinline__ int get_msb(uint32_t n) {
    return 31 - __clz(n | 1);
}

__device__ __forceinline__ int32_t log2_fixed(int32_t x) {
    if (x <= 0) return 0;
    int k = get_msb(x);
    int32_t frac = (k >= 4) ? ((x - (1 << k)) >> (k - 4)) : ((x - (1 << k)) << (4 - k));
    return (k << 4) + frac - 64;
}

// Fused embedding + LN
__global__ void __launch_bounds__(256)
fused_embed_ln_kernel(
    const uint8_t* __restrict__ data,
    long data_len,
    const int8_t* __restrict__ embedding,
    const int8_t* __restrict__ ln_w,
    float* __restrict__ X,
    const long* __restrict__ offsets,
    int t
) {
    int pop = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float smem[256];
    __shared__ float s_mean;
    __shared__ float warp_sums[8];
    
    uint8_t token = data[(offsets[pop] + t) % data_len];
    smem[tid] = (float)embedding[token * HIDDEN_DIM + tid];
    __syncthreads();
    
    float local_sum = fabsf(smem[tid]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    
    if (tid % 32 == 0) warp_sums[tid / 32] = local_sum;
    __syncthreads();
    
    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < 8; i++) sum += warp_sums[i];
        if (sum == 0) sum = 1;
        float mean = sum / HIDDEN_DIM;
        if (mean == 0) mean = 1;
        s_mean = mean;
    }
    __syncthreads();
    
    float val = smem[tid] * (float)ln_w[tid] / s_mean;
    X[pop * HIDDEN_DIM + tid] = clip_f(val);
}

// Fused layer norm
__global__ void __launch_bounds__(256)
fused_ln_kernel(float* __restrict__ X, const int8_t* __restrict__ ln_w) {
    int pop = blockIdx.x;
    int tid = threadIdx.x;
    
    float* x = X + pop * HIDDEN_DIM;
    
    __shared__ float s_mean;
    __shared__ float warp_sums[8];
    
    float local_sum = fabsf(x[tid]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    
    if (tid % 32 == 0) warp_sums[tid / 32] = local_sum;
    __syncthreads();
    
    if (tid == 0) {
        float sum = 0;
        for (int i = 0; i < 8; i++) sum += warp_sums[i];
        if (sum == 0) sum = 1;
        float mean = sum / HIDDEN_DIM;
        if (mean == 0) mean = 1;
        s_mean = mean;
    }
    __syncthreads();
    
    float val = x[tid] * (float)ln_w[tid] / s_mean;
    x[tid] = clip_f(val);
}

__global__ void __launch_bounds__(256)
fused_perturbation_kernel(
    float* __restrict__ out,      // [POPULATION_SIZE, out_dim]
    const float* __restrict__ X,  // [POPULATION_SIZE, in_dim]
    int out_dim,
    int in_dim,
    uint32_t seed
) {
    int pair = blockIdx.x;
    int tid = threadIdx.x;
    
    int pop_plus = pair * 2;
    int pop_minus = pair * 2 + 1;
    
    uint32_t pair_seed = seed + pair;
    
    __shared__ float warp_sums_plus[8];
    __shared__ float warp_sums_minus[8];
    __shared__ float s_xB_plus, s_xB_minus;
    
    const float* x_plus = X + pop_plus * in_dim;
    const float* x_minus = X + pop_minus * in_dim;
    
    float local_xB_plus = 0, local_xB_minus = 0;
    int vec_end = (in_dim / 4) * 4;
    for (int i = tid * 4; i < vec_end; i += blockDim.x * 4) {
        float4 xp = *reinterpret_cast<const float4*>(&x_plus[i]);
        float4 xm = *reinterpret_cast<const float4*>(&x_minus[i]);
        float b0 = (float)gen_noise_hash(pair_seed, out_dim + i);
        float b1 = (float)gen_noise_hash(pair_seed, out_dim + i + 1);
        float b2 = (float)gen_noise_hash(pair_seed, out_dim + i + 2);
        float b3 = (float)gen_noise_hash(pair_seed, out_dim + i + 3);
        local_xB_plus += xp.x * b0 + xp.y * b1 + xp.z * b2 + xp.w * b3;
        local_xB_minus += xm.x * b0 + xm.y * b1 + xm.z * b2 + xm.w * b3;
    }
    
    for (int i = vec_end + tid; i < in_dim; i += blockDim.x) {
        float b = (float)gen_noise_hash(pair_seed, out_dim + i);
        local_xB_plus += x_plus[i] * b;
        local_xB_minus += x_minus[i] * b;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_xB_plus += __shfl_down_sync(0xffffffff, local_xB_plus, offset);
        local_xB_minus += __shfl_down_sync(0xffffffff, local_xB_minus, offset);
    }
    
    if (tid % 32 == 0) {
        warp_sums_plus[tid / 32] = local_xB_plus;
        warp_sums_minus[tid / 32] = local_xB_minus;
    }
    __syncthreads();
    
    if (tid == 0) {
        float sum_plus = 0, sum_minus = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sum_plus += warp_sums_plus[i];
            sum_minus += warp_sums_minus[i];
        }
        s_xB_plus = sum_plus;
        s_xB_minus = sum_minus;
    }
    __syncthreads();
    
    float xB_plus = s_xB_plus;
    float xB_minus = s_xB_minus;
    float scale = 1.0f / (float)(1 << (FIXED_POINT + SIGMA_SHIFT));
    
    float* out_plus = out + pop_plus * out_dim;
    float* out_minus = out + pop_minus * out_dim;
    
    int out_vec_end = (out_dim / 4) * 4;
    for (int r = tid * 4; r < out_vec_end; r += blockDim.x * 4) {
        float4 op = *reinterpret_cast<float4*>(&out_plus[r]);
        float4 om = *reinterpret_cast<float4*>(&out_minus[r]);
        float a0 = (float)gen_noise_hash(pair_seed, r) * scale;
        float a1 = (float)gen_noise_hash(pair_seed, r + 1) * scale;
        float a2 = (float)gen_noise_hash(pair_seed, r + 2) * scale;
        float a3 = (float)gen_noise_hash(pair_seed, r + 3) * scale;
        op.x += xB_plus * a0; om.x -= xB_minus * a0;
        op.y += xB_plus * a1; om.y -= xB_minus * a1;
        op.z += xB_plus * a2; om.z -= xB_minus * a2;
        op.w += xB_plus * a3; om.w -= xB_minus * a3;
        *reinterpret_cast<float4*>(&out_plus[r]) = op;
        *reinterpret_cast<float4*>(&out_minus[r]) = om;
    }
    
    for (int r = out_vec_end + tid; r < out_dim; r += blockDim.x) {
        float a = (float)gen_noise_hash(pair_seed, r);
        float noise = a * scale;
        out_plus[r] += xB_plus * noise;
        out_minus[r] -= xB_minus * noise;
    }
}

__global__ void __launch_bounds__(256)
gru_gate_kernel(
    const float* __restrict__ buf1,
    const float* __restrict__ buf2,
    const int8_t* __restrict__ bias,
    const float* __restrict__ H,
    float* __restrict__ ft,
    float* __restrict__ gated_past
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POPULATION_SIZE * HIDDEN_DIM) return;
    
    int i = idx % HIDDEN_DIM;
    float sum = buf1[idx] + buf2[idx] + (float)bias[i];
    float f = clip_f(ldexpf(sum, -8));
    ft[idx] = f;
    gated_past[idx] = clip_f(((f + 127.0f) * H[idx]) / 256.0f);
}

__global__ void __launch_bounds__(256)
gru_ht_update_kernel(
    const float* __restrict__ buf1,
    const float* __restrict__ buf2,
    const int8_t* __restrict__ bias,
    const float* __restrict__ ft,
    float* __restrict__ H,
    float* __restrict__ X,
    const float* __restrict__ residual
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POPULATION_SIZE * HIDDEN_DIM) return;
    
    int i = idx % HIDDEN_DIM;
    float sum = buf1[idx] + buf2[idx] + (float)bias[i];
    float ht = clip_f(ldexpf(sum, -8));
    float h_old = H[idx];
    float update = ((ft[idx] + 127.0f) * (ht - h_old)) / 256.0f;
    float h_new = clip_f(h_old + update);
    H[idx] = h_new;
    X[idx] = clip_f(h_new + residual[idx]);
}

__global__ void __launch_bounds__(256)
mlp_residual_kernel(float* __restrict__ X, const float* __restrict__ residual, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= POPULATION_SIZE * HIDDEN_DIM) return;
    
    float val = clip_f(ldexpf(X[idx], -shift));
    X[idx] = clip_f(val + residual[idx]);
}

__global__ void copy_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ void __launch_bounds__(256)
loss_kernel(
    const float* __restrict__ logits,
    const uint8_t* __restrict__ data,
    long data_len,
    const long* __restrict__ offsets,
    int t,
    int32_t* __restrict__ losses
) {
    int pop = blockIdx.x;
    int tid = threadIdx.x;
    
    uint8_t target = data[(offsets[pop] + t + 1) % data_len];
    const float* log = logits + pop * VOCAB_SIZE;
    
    __shared__ int32_t warp_sums[8];
    
    int32_t local_sum = 0;
    for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
        int8_t logit = (int8_t)clip_f(ldexpf(log[i], -8));
        int idx = (int32_t)logit + 128;
        idx = max(0, min(255, idx));
        local_sum += d_EXP2_TABLE[idx];
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    
    if (tid % 32 == 0) warp_sums[tid / 32] = local_sum;
    __syncthreads();
    
    if (tid == 0) {
        int32_t sum_exp = 0;
        for (int i = 0; i < 8; i++) sum_exp += warp_sums[i];
        int32_t log_sum = log2_fixed(sum_exp);
        int8_t target_logit = (int8_t)clip_f(ldexpf(log[target], -8));
        int32_t target_val = (int32_t)target_logit + 128;
        atomicAdd(&losses[pop], log_sum - target_val);
    }
}

__global__ void __launch_bounds__(256)
update_weights_kernel(int8_t* W, int out_dim, int in_dim, uint32_t seed, const int* __restrict__ fitnesses, int pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim * in_dim) return;
    
    int o = idx / in_dim;
    int i = idx % in_dim;
    
    int32_t vote = 0;
    for (int p = 0; p < pairs; p++) {
        int f = fitnesses[p];
        if (f == 0) continue;
        
        uint32_t pair_seed = seed + p;
        int8_t a = gen_noise_hash(pair_seed, o);
        int8_t b = gen_noise_hash(pair_seed, out_dim + i);
        
        vote += (int32_t)(a * f) * (int32_t)b;
    }
    
    int8_t w = W[idx];
    if (vote > UPDATE_THRESHOLD && w < MAX_VAL) W[idx] = w + 1;
    else if (vote < -UPDATE_THRESHOLD && w > MIN_VAL) W[idx] = w - 1;
}

__global__ void int8_to_float_kernel(const int8_t* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}


struct EggModel {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM];
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * MLP_EXPAND_DIM];
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
};

class Trainer {
public:
    cublasHandle_t cublas;
    
    int8_t *d_emb, *d_gru_w, *d_gru_b, *d_mlp_w, *d_head, *d_ln_w, *d_ln_out;
    float *d_gru_wf[N_LAYERS][4], *d_mlp_wf[N_LAYERS][2], *d_headf;
    float *d_X, *d_residual, *d_buf1, *d_buf2, *d_ft, *d_gated_past;
    float *d_H[N_LAYERS];
    float *d_logits;
    long *d_offsets;
    int *d_fit;
    int32_t *d_losses;
    uint8_t *d_data;
    long data_len;
    
#ifdef USE_INT8_TC
    // INT8 tensor core buffers
    int8_t *d_X_i8, *d_H_i8[N_LAYERS], *d_buf1_i8, *d_buf2_i8;
    int32_t *d_gemm_out;  // INT32 GEMM output buffer
#endif
    
    Trainer() {
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH));
        
        CUDA_CHECK(cudaMalloc(&d_emb, VOCAB_SIZE * HIDDEN_DIM));
        CUDA_CHECK(cudaMalloc(&d_gru_w, N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM));
        CUDA_CHECK(cudaMalloc(&d_gru_b, N_LAYERS * 2 * HIDDEN_DIM));
        CUDA_CHECK(cudaMalloc(&d_mlp_w, N_LAYERS * 2 * HIDDEN_DIM * MLP_EXPAND_DIM));
        CUDA_CHECK(cudaMalloc(&d_head, HIDDEN_DIM * VOCAB_SIZE));
        CUDA_CHECK(cudaMalloc(&d_ln_w, N_LAYERS * 2 * HIDDEN_DIM));
        CUDA_CHECK(cudaMalloc(&d_ln_out, HIDDEN_DIM));
        
        for (int l = 0; l < N_LAYERS; l++) {
            for (int w = 0; w < 4; w++)
                CUDA_CHECK(cudaMalloc(&d_gru_wf[l][w], HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
            for (int w = 0; w < 2; w++)
                CUDA_CHECK(cudaMalloc(&d_mlp_wf[l][w], HIDDEN_DIM * MLP_EXPAND_DIM * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_headf, HIDDEN_DIM * VOCAB_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_X, POPULATION_SIZE * MLP_EXPAND_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_residual, POPULATION_SIZE * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buf1, POPULATION_SIZE * MLP_EXPAND_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_buf2, POPULATION_SIZE * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ft, POPULATION_SIZE * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gated_past, POPULATION_SIZE * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_logits, POPULATION_SIZE * VOCAB_SIZE * sizeof(float)));
        for (int l = 0; l < N_LAYERS; l++)
            CUDA_CHECK(cudaMalloc(&d_H[l], POPULATION_SIZE * HIDDEN_DIM * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_offsets, POPULATION_SIZE * sizeof(long)));
        CUDA_CHECK(cudaMalloc(&d_fit, PAIRS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_losses, POPULATION_SIZE * sizeof(int32_t)));
        
#ifdef USE_INT8_TC
        // Allocate INT8 buffers for tensor core path
        CUDA_CHECK(cudaMalloc(&d_X_i8, POPULATION_SIZE * MLP_EXPAND_DIM));
        CUDA_CHECK(cudaMalloc(&d_buf1_i8, POPULATION_SIZE * MLP_EXPAND_DIM));
        CUDA_CHECK(cudaMalloc(&d_buf2_i8, POPULATION_SIZE * HIDDEN_DIM));
        CUDA_CHECK(cudaMalloc(&d_gemm_out, POPULATION_SIZE * MLP_EXPAND_DIM * sizeof(int32_t)));
        for (int l = 0; l < N_LAYERS; l++)
            CUDA_CHECK(cudaMalloc(&d_H_i8[l], POPULATION_SIZE * HIDDEN_DIM));
#endif
        
        d_data = nullptr;
    }
    
    ~Trainer() {
        cublasDestroy(cublas);
        cudaFree(d_emb); cudaFree(d_gru_w); cudaFree(d_gru_b);
        cudaFree(d_mlp_w); cudaFree(d_head); cudaFree(d_ln_w); cudaFree(d_ln_out);
        for (int l = 0; l < N_LAYERS; l++) {
            for (int w = 0; w < 4; w++) cudaFree(d_gru_wf[l][w]);
            for (int w = 0; w < 2; w++) cudaFree(d_mlp_wf[l][w]);
            cudaFree(d_H[l]);
        }
        cudaFree(d_headf);
        cudaFree(d_X); cudaFree(d_residual); cudaFree(d_buf1); cudaFree(d_buf2);
        cudaFree(d_ft); cudaFree(d_gated_past); cudaFree(d_logits);
        cudaFree(d_offsets); cudaFree(d_fit); cudaFree(d_losses);
        if (d_data) cudaFree(d_data);
#ifdef USE_INT8_TC
        cudaFree(d_X_i8); cudaFree(d_buf1_i8); cudaFree(d_buf2_i8); cudaFree(d_gemm_out);
        for (int l = 0; l < N_LAYERS; l++) cudaFree(d_H_i8[l]);
#endif
    }
    
    void upload(const EggModel* m) {
        CUDA_CHECK(cudaMemcpy(d_emb, m->embedding, VOCAB_SIZE * HIDDEN_DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gru_w, m->gru_weights, N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gru_b, m->gru_biases, N_LAYERS * 2 * HIDDEN_DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mlp_w, m->mlp_weights, N_LAYERS * 2 * HIDDEN_DIM * MLP_EXPAND_DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_head, m->head, HIDDEN_DIM * VOCAB_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ln_w, m->ln_weights, N_LAYERS * 2 * HIDDEN_DIM, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ln_out, m->ln_out, HIDDEN_DIM, cudaMemcpyHostToDevice));
        
        for (int l = 0; l < N_LAYERS; l++) {
            for (int w = 0; w < 4; w++) {
                int8_t* src = d_gru_w + l * 4 * HIDDEN_DIM * HIDDEN_DIM + w * HIDDEN_DIM * HIDDEN_DIM;
                int8_to_float_kernel<<<(HIDDEN_DIM*HIDDEN_DIM+255)/256, 256>>>(src, d_gru_wf[l][w], HIDDEN_DIM * HIDDEN_DIM);
            }
            for (int w = 0; w < 2; w++) {
                int8_t* src = d_mlp_w + l * 2 * HIDDEN_DIM * MLP_EXPAND_DIM + w * HIDDEN_DIM * MLP_EXPAND_DIM;
                int8_to_float_kernel<<<(HIDDEN_DIM*MLP_EXPAND_DIM+255)/256, 256>>>(src, d_mlp_wf[l][w], HIDDEN_DIM * MLP_EXPAND_DIM);
            }
        }
        int8_to_float_kernel<<<(HIDDEN_DIM*VOCAB_SIZE+255)/256, 256>>>(d_head, d_headf, HIDDEN_DIM * VOCAB_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void upload_data(const uint8_t* data, long len) {
        data_len = len;
        CUDA_CHECK(cudaMalloc(&d_data, len));
        CUDA_CHECK(cudaMemcpy(d_data, data, len, cudaMemcpyHostToDevice));
    }
    
#ifdef USE_INT8_TC
    void int8_gemm_helper(
        float* out, const float* X, const int8_t* W,
        int batch, int out_dim, int in_dim,
        int8_t* X_i8_buf, int32_t* gemm_buf
    ) {
        int total_in = batch * in_dim;
        int total_out = batch * out_dim;
        
        quantize_kernel<<<(total_in + 255) / 256, 256>>>(X_i8_buf, X, total_in);
        int8_gemm_tn(gemm_buf, W, X_i8_buf, out_dim, batch, in_dim);
        dequantize_kernel<<<(total_out + 255) / 256, 256>>>(out, gemm_buf, total_out);
    }
#endif
    
    void train_step(long start_idx, uint32_t step_seed, int* h_fit, int32_t* h_loss) {
        long stride = data_len / PAIRS;
        
        long h_offsets[POPULATION_SIZE];
        for (int p = 0; p < PAIRS; p++) {
            long offset = (start_idx + p * stride) % (data_len - SEQ_LEN - 1);
            h_offsets[p * 2] = offset;
            h_offsets[p * 2 + 1] = offset;
        }
        CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets, POPULATION_SIZE * sizeof(long), cudaMemcpyHostToDevice));
        
        for (int l = 0; l < N_LAYERS; l++)
            CUDA_CHECK(cudaMemsetAsync(d_H[l], 0, POPULATION_SIZE * HIDDEN_DIM * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_losses, 0, POPULATION_SIZE * sizeof(int32_t)));
        
        int total_hd = POPULATION_SIZE * HIDDEN_DIM;
#ifndef USE_INT8_TC
        float alpha = 1.0f, beta = 0.0f;
#endif
        
        for (int t = 0; t < SEQ_LEN; t++) {
            const int8_t* ln_w0 = d_ln_w;
            fused_embed_ln_kernel<<<POPULATION_SIZE, HIDDEN_DIM>>>(d_data, data_len, d_emb, ln_w0, d_X, d_offsets, t);
            
            for (int l = 0; l < N_LAYERS; l++) {
                uint32_t l_seed = step_seed + l * 1000;
                const int8_t* bias_f = d_gru_b + l * 2 * HIDDEN_DIM;
                const int8_t* bias_h = bias_f + HIDDEN_DIM;
                const int8_t* ln_w1 = d_ln_w + l * 2 * HIDDEN_DIM + HIDDEN_DIM;
                
                CUDA_CHECK(cudaMemcpyAsync(d_residual, d_X, total_hd * sizeof(float), cudaMemcpyDeviceToDevice));
                
                if (l > 0) {
                    const int8_t* ln_w0_l = d_ln_w + l * 2 * HIDDEN_DIM;
                    fused_ln_kernel<<<POPULATION_SIZE, HIDDEN_DIM>>>(d_X, ln_w0_l);
                }
                
#ifdef USE_INT8_TC
                const int8_t* gru_w0 = d_gru_w + l * 4 * HIDDEN_DIM * HIDDEN_DIM;
                const int8_t* gru_w1 = gru_w0 + HIDDEN_DIM * HIDDEN_DIM;
                const int8_t* gru_w2 = gru_w1 + HIDDEN_DIM * HIDDEN_DIM;
                const int8_t* gru_w3 = gru_w2 + HIDDEN_DIM * HIDDEN_DIM;
                const int8_t* mlp_w0 = d_mlp_w + l * 2 * HIDDEN_DIM * MLP_EXPAND_DIM;
                const int8_t* mlp_w1 = mlp_w0 + HIDDEN_DIM * MLP_EXPAND_DIM;
                
                int8_gemm_helper(d_buf1, d_X, gru_w0, POPULATION_SIZE, HIDDEN_DIM, HIDDEN_DIM, d_X_i8, d_gemm_out);
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf1, d_X, HIDDEN_DIM, HIDDEN_DIM, l_seed+1);
                
                int8_gemm_helper(d_buf2, d_H[l], gru_w1, POPULATION_SIZE, HIDDEN_DIM, HIDDEN_DIM, d_X_i8, d_gemm_out);
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf2, d_H[l], HIDDEN_DIM, HIDDEN_DIM, l_seed+2);
                
                gru_gate_kernel<<<(total_hd+255)/256, 256>>>(d_buf1, d_buf2, bias_f, d_H[l], d_ft, d_gated_past);
                
                int8_gemm_helper(d_buf1, d_X, gru_w2, POPULATION_SIZE, HIDDEN_DIM, HIDDEN_DIM, d_X_i8, d_gemm_out);
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf1, d_X, HIDDEN_DIM, HIDDEN_DIM, l_seed+3);
                
                int8_gemm_helper(d_buf2, d_gated_past, gru_w3, POPULATION_SIZE, HIDDEN_DIM, HIDDEN_DIM, d_X_i8, d_gemm_out);
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf2, d_gated_past, HIDDEN_DIM, HIDDEN_DIM, l_seed+4);
                
                gru_ht_update_kernel<<<(total_hd+255)/256, 256>>>(d_buf1, d_buf2, bias_h, d_ft, d_H[l], d_X, d_residual);
                
                CUDA_CHECK(cudaMemcpyAsync(d_residual, d_X, total_hd * sizeof(float), cudaMemcpyDeviceToDevice));
                fused_ln_kernel<<<POPULATION_SIZE, HIDDEN_DIM>>>(d_X, ln_w1);
                
                int8_gemm_helper(d_buf1, d_X, mlp_w0, POPULATION_SIZE, MLP_EXPAND_DIM, HIDDEN_DIM, d_X_i8, d_gemm_out);
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf1, d_X, MLP_EXPAND_DIM, HIDDEN_DIM, l_seed+5);
                
                int8_gemm_helper(d_X, d_buf1, mlp_w1, POPULATION_SIZE, HIDDEN_DIM, MLP_EXPAND_DIM, d_buf1_i8, d_gemm_out);
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_X, d_buf1, HIDDEN_DIM, MLP_EXPAND_DIM, l_seed+6);
                
                mlp_residual_kernel<<<(total_hd+255)/256, 256>>>(d_X, d_residual, 17);
#else
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_DIM, POPULATION_SIZE, HIDDEN_DIM, &alpha,
                    d_gru_wf[l][0], HIDDEN_DIM, d_X, HIDDEN_DIM, &beta, d_buf1, HIDDEN_DIM));
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf1, d_X, HIDDEN_DIM, HIDDEN_DIM, l_seed+1);
                
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_DIM, POPULATION_SIZE, HIDDEN_DIM, &alpha,
                    d_gru_wf[l][1], HIDDEN_DIM, d_H[l], HIDDEN_DIM, &beta, d_buf2, HIDDEN_DIM));
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf2, d_H[l], HIDDEN_DIM, HIDDEN_DIM, l_seed+2);
                
                gru_gate_kernel<<<(total_hd+255)/256, 256>>>(d_buf1, d_buf2, bias_f, d_H[l], d_ft, d_gated_past);
                
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_DIM, POPULATION_SIZE, HIDDEN_DIM, &alpha,
                    d_gru_wf[l][2], HIDDEN_DIM, d_X, HIDDEN_DIM, &beta, d_buf1, HIDDEN_DIM));
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf1, d_X, HIDDEN_DIM, HIDDEN_DIM, l_seed+3);
                
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_DIM, POPULATION_SIZE, HIDDEN_DIM, &alpha,
                    d_gru_wf[l][3], HIDDEN_DIM, d_gated_past, HIDDEN_DIM, &beta, d_buf2, HIDDEN_DIM));
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf2, d_gated_past, HIDDEN_DIM, HIDDEN_DIM, l_seed+4);
                
                gru_ht_update_kernel<<<(total_hd+255)/256, 256>>>(d_buf1, d_buf2, bias_h, d_ft, d_H[l], d_X, d_residual);
                
                CUDA_CHECK(cudaMemcpyAsync(d_residual, d_X, total_hd * sizeof(float), cudaMemcpyDeviceToDevice));
                fused_ln_kernel<<<POPULATION_SIZE, HIDDEN_DIM>>>(d_X, ln_w1);
                
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    MLP_EXPAND_DIM, POPULATION_SIZE, HIDDEN_DIM, &alpha,
                    d_mlp_wf[l][0], HIDDEN_DIM, d_X, HIDDEN_DIM, &beta, d_buf1, MLP_EXPAND_DIM));
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_buf1, d_X, MLP_EXPAND_DIM, HIDDEN_DIM, l_seed+5);
                
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    HIDDEN_DIM, POPULATION_SIZE, MLP_EXPAND_DIM, &alpha,
                    d_mlp_wf[l][1], MLP_EXPAND_DIM, d_buf1, MLP_EXPAND_DIM, &beta, d_X, HIDDEN_DIM));
                fused_perturbation_kernel<<<PAIRS, 256>>>(d_X, d_buf1, HIDDEN_DIM, MLP_EXPAND_DIM, l_seed+6);
                
                mlp_residual_kernel<<<(total_hd+255)/256, 256>>>(d_X, d_residual, 17);
#endif
            }
            
            // Output head
            fused_ln_kernel<<<POPULATION_SIZE, HIDDEN_DIM>>>(d_X, d_ln_out);
#ifdef USE_INT8_TC
            int8_gemm_helper(d_logits, d_X, d_head, POPULATION_SIZE, VOCAB_SIZE, HIDDEN_DIM,
                             d_X_i8, d_gemm_out);
#else
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                VOCAB_SIZE, POPULATION_SIZE, HIDDEN_DIM, &alpha,
                d_headf, HIDDEN_DIM, d_X, HIDDEN_DIM, &beta, d_logits, VOCAB_SIZE));
#endif
            fused_perturbation_kernel<<<PAIRS, 256>>>(d_logits, d_X, VOCAB_SIZE, HIDDEN_DIM, step_seed+999);
            
            loss_kernel<<<POPULATION_SIZE, 256>>>(d_logits, d_data, data_len, d_offsets, t, d_losses);
        }
        
        // cudaMemcpy is synchronous, no need for explicit sync
        CUDA_CHECK(cudaMemcpy(h_loss, d_losses, POPULATION_SIZE * sizeof(int32_t), cudaMemcpyDeviceToHost));
        
        for (int p = 0; p < PAIRS; p++) {
            if (h_loss[p*2] < h_loss[p*2+1]) h_fit[p] = 1;
            else if (h_loss[p*2+1] < h_loss[p*2]) h_fit[p] = -1;
            else h_fit[p] = 0;
        }
        
        // Check if any fitness is non-zero (skip update if all ties)
        bool any_nonzero = false;
        for (int p = 0; p < PAIRS; p++) {
            if (h_fit[p] != 0) { any_nonzero = true; break; }
        }
        
        if (!any_nonzero) return;  // Skip weight updates if all ties
        
        CUDA_CHECK(cudaMemcpy(d_fit, h_fit, PAIRS * sizeof(int), cudaMemcpyHostToDevice));
        
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t l_seed = step_seed + l * 1000;
            for (int w = 0; w < 4; w++) {
                int8_t* gru_w = d_gru_w + l * 4 * HIDDEN_DIM * HIDDEN_DIM + w * HIDDEN_DIM * HIDDEN_DIM;
                // GRU: out_dim=HIDDEN_DIM, in_dim=HIDDEN_DIM
                update_weights_kernel<<<(HIDDEN_DIM*HIDDEN_DIM+255)/256, 256>>>(
                    gru_w, HIDDEN_DIM, HIDDEN_DIM, l_seed+1+w, d_fit, PAIRS);
            }
            int8_t* mlp_w0 = d_mlp_w + l * 2 * HIDDEN_DIM * MLP_EXPAND_DIM;
            int8_t* mlp_w1 = mlp_w0 + HIDDEN_DIM * MLP_EXPAND_DIM;
            // MLP expand: out_dim=MLP_EXPAND_DIM, in_dim=HIDDEN_DIM
            update_weights_kernel<<<(MLP_EXPAND_DIM*HIDDEN_DIM+255)/256, 256>>>(
                mlp_w0, MLP_EXPAND_DIM, HIDDEN_DIM, l_seed+5, d_fit, PAIRS);
            // MLP project: out_dim=HIDDEN_DIM, in_dim=MLP_EXPAND_DIM
            update_weights_kernel<<<(HIDDEN_DIM*MLP_EXPAND_DIM+255)/256, 256>>>(
                mlp_w1, HIDDEN_DIM, MLP_EXPAND_DIM, l_seed+6, d_fit, PAIRS);
        }
        // Head: out_dim=VOCAB_SIZE, in_dim=HIDDEN_DIM
        update_weights_kernel<<<(VOCAB_SIZE*HIDDEN_DIM+255)/256, 256>>>(
            d_head, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, d_fit, PAIRS);
        
        for (int l = 0; l < N_LAYERS; l++) {
            for (int w = 0; w < 4; w++) {
                int8_t* src = d_gru_w + l * 4 * HIDDEN_DIM * HIDDEN_DIM + w * HIDDEN_DIM * HIDDEN_DIM;
                int8_to_float_kernel<<<(HIDDEN_DIM*HIDDEN_DIM+255)/256, 256>>>(src, d_gru_wf[l][w], HIDDEN_DIM * HIDDEN_DIM);
            }
            for (int w = 0; w < 2; w++) {
                int8_t* src = d_mlp_w + l * 2 * HIDDEN_DIM * MLP_EXPAND_DIM + w * HIDDEN_DIM * MLP_EXPAND_DIM;
                int8_to_float_kernel<<<(HIDDEN_DIM*MLP_EXPAND_DIM+255)/256, 256>>>(src, d_mlp_wf[l][w], HIDDEN_DIM * MLP_EXPAND_DIM);
            }
        }
        int8_to_float_kernel<<<(HIDDEN_DIM*VOCAB_SIZE+255)/256, 256>>>(d_head, d_headf, HIDDEN_DIM * VOCAB_SIZE);
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

static inline uint32_t h_xorshift32(uint32_t* s) {
    uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x; return x;
}

static inline int8_t h_gen_noise(uint32_t* r) {
    uint32_t v = h_xorshift32(r);
    return (int8_t)((v & 1 ? 1 : -1) * ((v >> 1) & 31));
}

void init_model(EggModel* m, uint32_t seed) {
    uint32_t rng = seed;
    for (int i = 0; i < VOCAB_SIZE * HIDDEN_DIM; i++) m->embedding[i] = h_gen_noise(&rng);
    for (int i = 0; i < HIDDEN_DIM * VOCAB_SIZE; i++) m->head[i] = h_gen_noise(&rng);
    for (int l = 0; l < N_LAYERS; l++) {
        for (int g = 0; g < 4; g++)
            for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++)
                m->gru_weights[l][g][i] = h_gen_noise(&rng);
        for (int w = 0; w < 2; w++)
            for (int i = 0; i < HIDDEN_DIM * MLP_EXPAND_DIM; i++)
                m->mlp_weights[l][w][i] = h_gen_noise(&rng);
        memset(m->gru_biases[l], 0, sizeof(m->gru_biases[l]));
        for (int i = 0; i < HIDDEN_DIM; i++) {
            m->ln_weights[l][0][i] = 16;
            m->ln_weights[l][1][i] = 16;
        }
    }
    for (int i = 0; i < HIDDEN_DIM; i++) m->ln_out[i] = 16;
}

void init_tables() {
    int32_t h_EXP2_TABLE[256];
    for (int i = 0; i < 256; i++)
        h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / 16.0) * 16.0);
    CUDA_CHECK(cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, sizeof(h_EXP2_TABLE)));
}

int main() {
    srand(time(NULL));
    init_tables();
    
    FILE* f = fopen("input.txt", "rb");
    if (!f) { printf("Error: input.txt not found\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(len);
    size_t r = fread(data, 1, len, f);
    (void)r;
    fclose(f);
    printf("Loaded: %ld bytes\n", len);
    fflush(stdout);
    
    EggModel* m = (EggModel*)aligned_alloc(64, sizeof(EggModel));
    init_model(m, 42);
    
    Trainer tr;
    tr.upload(m);
    tr.upload_data(data, len);
    
    int* fit = (int*)malloc(PAIRS * sizeof(int));
    int32_t* loss = (int32_t*)malloc(POPULATION_SIZE * sizeof(int32_t));
    
#ifdef USE_INT8_TC
    printf("Training: H=%d L=%d S=%d P=%d (INT8 Tensor Cores)\n", HIDDEN_DIM, N_LAYERS, SEQ_LEN, POPULATION_SIZE);
    print_int8_tc_info();
#else
    printf("Training: H=%d L=%d S=%d P=%d (cuBLAS TF32)\n", HIDDEN_DIM, N_LAYERS, SEQ_LEN, POPULATION_SIZE);
#endif
    fflush(stdout);
    
    long max_steps = (len - 1) / SEQ_LEN;
    
    for (long step = 0; step < max_steps; step++) {
        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        long idx = step * SEQ_LEN;
        
        struct timespec s0;
        clock_gettime(CLOCK_MONOTONIC, &s0);
        
        tr.train_step(idx, seed, fit, loss);
        
        struct timespec s1;
        clock_gettime(CLOCK_MONOTONIC, &s1);
        double ms = (s1.tv_sec - s0.tv_sec) * 1000.0 + (s1.tv_nsec - s0.tv_nsec) / 1e6;
        
        int32_t avg_loss = 0;
        for (int i = 0; i < POPULATION_SIZE; i++) avg_loss += loss[i];
        avg_loss /= POPULATION_SIZE;
        float loss_val = (float)avg_loss / (SEQ_LEN * 16.0f);
        
        int pos_better = 0, neg_better = 0;
        for (int i = 0; i < PAIRS; i++) {
            if (fit[i] == 1) pos_better++;
            else if (fit[i] == -1) neg_better++;
        }
        
        printf("\033[32m");
        for (int i = 0; i < 30 && idx + i < len; i++) {
            char c = data[idx + i];
            printf("%c", (c >= 32 && c <= 126) ? c : '.');
        }
        printf("\033[36m..............................\033[0m\n");
        printf("Step %ld | Loss: %.4f | +Perturb Lower Loss: %d | -Perturb Lower Loss: %d | %.1f ms\n",
               step, loss_val, pos_better, neg_better, ms);
        fflush(stdout);
    }
    
    printf("Done.\n");
    free(data);
    free(m);
    free(fit);
    free(loss);
    return 0;
}
