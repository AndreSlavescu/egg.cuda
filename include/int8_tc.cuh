#ifndef INT8_TC_CUH
#define INT8_TC_CUH

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>
#include <unordered_map>

namespace cg = cooperative_groups;

struct GemmKey {
    int M, N, K;
    bool operator==(const GemmKey& o) const { return M == o.M && N == o.N && K == o.K; }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        return ((size_t)k.M << 40) | ((size_t)k.N << 20) | k.K;
    }
};

struct CachedGemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t layoutW, layoutX, layoutC;
    cublasLtMatmulAlgo_t algo;
    bool hasAlgo;
};

class CublasLtInt8Gemm {
public:
    cublasLtHandle_t handle;
    void* workspace;
    size_t workspaceSize;
    cublasLtMatmulPreference_t pref;
    std::unordered_map<GemmKey, CachedGemm, GemmKeyHash> cache;
    
    CublasLtInt8Gemm() : handle(nullptr), workspace(nullptr), workspaceSize(0) {
        cublasLtCreate(&handle);
        workspaceSize = 32 * 1024 * 1024;
        cudaMalloc(&workspace, workspaceSize);
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    }
    
    ~CublasLtInt8Gemm() {
        for (auto& kv : cache) {
            cublasLtMatrixLayoutDestroy(kv.second.layoutW);
            cublasLtMatrixLayoutDestroy(kv.second.layoutX);
            cublasLtMatrixLayoutDestroy(kv.second.layoutC);
            cublasLtMatmulDescDestroy(kv.second.desc);
        }
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (workspace) cudaFree(workspace);
        if (handle) cublasLtDestroy(handle);
    }
    
    CachedGemm& getOrCreate(int N, int M, int K) {
        GemmKey key{M, N, K};
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        
        CachedGemm& c = cache[key];
        
        cublasLtMatmulDescCreate(&c.desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(c.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(c.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        
        cublasLtMatrixLayoutCreate(&c.layoutW, CUDA_R_8I, K, N, K);
        cublasLtMatrixLayoutCreate(&c.layoutX, CUDA_R_8I, K, M, K);
        cublasLtMatrixLayoutCreate(&c.layoutC, CUDA_R_32I, N, M, N);
        
        cublasLtMatmulHeuristicResult_t heur;
        int found = 0;
        cublasLtMatmulAlgoGetHeuristic(handle, c.desc, c.layoutW, c.layoutX, c.layoutC, c.layoutC, pref, 1, &heur, &found);
        c.hasAlgo = (found > 0);
        if (c.hasAlgo) c.algo = heur.algo;
        
        return c;
    }
    
    void gemm_tn(int32_t* C, const int8_t* W, const int8_t* X, int N, int M, int K, cudaStream_t stream = 0) {
        CachedGemm& c = getOrCreate(N, M, K);
        int32_t alpha = 1, beta = 0;
        cublasLtMatmul(handle, c.desc, &alpha, W, c.layoutW, X, c.layoutX, &beta, C, c.layoutC, C, c.layoutC,
                       c.hasAlgo ? &c.algo : nullptr, workspace, workspaceSize, stream);
    }
};

static CublasLtInt8Gemm* g_cublaslt_int8 = nullptr;

inline CublasLtInt8Gemm* get_cublaslt_int8() {
    if (!g_cublaslt_int8) g_cublaslt_int8 = new CublasLtInt8Gemm();
    return g_cublaslt_int8;
}

inline void int8_gemm_tn(int32_t* C, const int8_t* W, const int8_t* X, int N, int M, int K, cudaStream_t stream = 0) {
    get_cublaslt_int8()->gemm_tn(C, W, X, N, M, K, stream);
}

__global__ void __launch_bounds__(256, 4)
quantize_kernel(int8_t* __restrict__ out, const float* __restrict__ in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (int8_t)__float2int_rn(fminf(fmaxf(in[i], -127.0f), 127.0f));
    }
}

__device__ __forceinline__ uint32_t perturb_hash_rng(uint32_t seed, uint32_t idx) {
    uint32_t x = seed + idx * 0x9e3779b9;
    x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t perturb_gen_noise(uint32_t seed, uint32_t idx) {
    uint32_t r = perturb_hash_rng(seed, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__global__ void __launch_bounds__(256, 2)
fused_dequant_perturb_kernel(
    float* __restrict__ out,
    const int32_t* __restrict__ gemm_out,
    const int8_t* __restrict__ X_i8,
    int out_dim,
    int in_dim,
    int population_size,
    uint32_t seed
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int pair = blockIdx.x;
    int tid = threadIdx.x;
    
    int pop_plus = pair * 2;
    int pop_minus = pair * 2 + 1;
    uint32_t pair_seed = seed + pair;
    
    __shared__ float warp_sums_plus[8];
    __shared__ float warp_sums_minus[8];
    __shared__ float s_xB_plus, s_xB_minus;
    
    const int8_t* x_plus = X_i8 + pop_plus * in_dim;
    const int8_t* x_minus = X_i8 + pop_minus * in_dim;
    
    float local_xB_plus = 0, local_xB_minus = 0;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        float b = (float)perturb_gen_noise(pair_seed, out_dim + i);
        local_xB_plus += (float)x_plus[i] * b;
        local_xB_minus += (float)x_minus[i] * b;
    }
    
    float warp_sum_plus = cg::reduce(warp, local_xB_plus, cg::plus<float>());
    float warp_sum_minus = cg::reduce(warp, local_xB_minus, cg::plus<float>());
    
    if (warp.thread_rank() == 0) {
        warp_sums_plus[tid / 32] = warp_sum_plus;
        warp_sums_minus[tid / 32] = warp_sum_minus;
    }
    block.sync();
    
    if (tid == 0) {
        float sum_plus = 0, sum_minus = 0;
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            sum_plus += warp_sums_plus[i];
            sum_minus += warp_sums_minus[i];
        }
        s_xB_plus = sum_plus;
        s_xB_minus = sum_minus;
    }
    block.sync();
    
    float xB_plus = s_xB_plus;
    float xB_minus = s_xB_minus;
    float perturb_scale = 1.0f / 256.0f;
    float gemm_scale = 1.0f / 256.0f;
    
    const int32_t* gemm_plus = gemm_out + pop_plus * out_dim;
    const int32_t* gemm_minus = gemm_out + pop_minus * out_dim;
    float* out_plus = out + pop_plus * out_dim;
    float* out_minus = out + pop_minus * out_dim;
    
    for (int r = tid; r < out_dim; r += blockDim.x) {
        float a = (float)perturb_gen_noise(pair_seed, r) * perturb_scale;
        out_plus[r] = (float)gemm_plus[r] * gemm_scale + xB_plus * a;
        out_minus[r] = (float)gemm_minus[r] * gemm_scale - xB_minus * a;
    }
}

__global__ void __launch_bounds__(256, 2)
fused_i32_perturb_to_i8_kernel(
    int8_t* __restrict__ out_i8,
    const int32_t* __restrict__ gemm_out,
    const int8_t* __restrict__ X_i8,
    int out_dim,
    int in_dim,
    int population_size,
    uint32_t seed,
    int shift
) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    int pair = blockIdx.x;
    int tid = threadIdx.x;
    
    int pop_plus = pair * 2;
    int pop_minus = pair * 2 + 1;
    uint32_t pair_seed = seed + pair;
    
    __shared__ int warp_sums_plus[8];
    __shared__ int warp_sums_minus[8];
    __shared__ int s_xB_plus, s_xB_minus;
    
    const int8_t* x_plus = X_i8 + pop_plus * in_dim;
    const int8_t* x_minus = X_i8 + pop_minus * in_dim;
    
    int local_xB_plus = 0, local_xB_minus = 0;
    for (int i = tid; i < in_dim; i += blockDim.x) {
        int b = (int)perturb_gen_noise(pair_seed, out_dim + i);
        local_xB_plus += (int)x_plus[i] * b;
        local_xB_minus += (int)x_minus[i] * b;
    }
    
    int warp_sum_plus = cg::reduce(warp, local_xB_plus, cg::plus<int>());
    int warp_sum_minus = cg::reduce(warp, local_xB_minus, cg::plus<int>());
    
    if (warp.thread_rank() == 0) {
        warp_sums_plus[tid / 32] = warp_sum_plus;
        warp_sums_minus[tid / 32] = warp_sum_minus;
    }
    block.sync();
    
    if (tid == 0) {
        int sum_plus = 0, sum_minus = 0;
        #pragma unroll 8
        for (int i = 0; i < 8; i++) {
            sum_plus += warp_sums_plus[i];
            sum_minus += warp_sums_minus[i];
        }
        s_xB_plus = sum_plus;
        s_xB_minus = sum_minus;
    }
    block.sync();
    
    int xB_plus = s_xB_plus;
    int xB_minus = s_xB_minus;
    
    const int32_t* gemm_plus = gemm_out + pop_plus * out_dim;
    const int32_t* gemm_minus = gemm_out + pop_minus * out_dim;
    int8_t* out_i8_plus = out_i8 + pop_plus * out_dim;
    int8_t* out_i8_minus = out_i8 + pop_minus * out_dim;
    
    for (int r = tid; r < out_dim; r += blockDim.x) {
        int a = (int)perturb_gen_noise(pair_seed, r);
        int perturb_plus = (xB_plus * a) >> 8;
        int perturb_minus = (xB_minus * a) >> 8;
        
        int val_plus = (gemm_plus[r] >> shift) + perturb_plus;
        int val_minus = (gemm_minus[r] >> shift) - perturb_minus;
        
        val_plus = max(-127, min(127, val_plus));
        val_minus = max(-127, min(127, val_minus));
        
        out_i8_plus[r] = (int8_t)val_plus;
        out_i8_minus[r] = (int8_t)val_minus;
    }
}

__global__ void __launch_bounds__(256)
fused_i32_simple_perturb_to_f32_kernel(
    float* __restrict__ out_f32,
    const int32_t* __restrict__ gemm_out,
    int out_dim,
    int population_size,
    uint32_t seed,
    int shift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = population_size * out_dim;
    if (idx >= total) return;
    
    int pop = idx / out_dim;
    int r = idx % out_dim;
    int pair = pop / 2;
    int is_minus = pop & 1;
    uint32_t pair_seed = seed + pair;
    
    float dequant_scale = ldexpf(1.0f, -shift);
    float val = (float)gemm_out[idx] * dequant_scale;
    
    float noise = (float)perturb_gen_noise(pair_seed, r) * 0.25f;
    val += is_minus ? -noise : noise;
    
    out_f32[idx] = val;
}

__global__ void __launch_bounds__(256)
fused_i32_simple_perturb_to_i8_kernel(
    int8_t* __restrict__ out_i8,
    const int32_t* __restrict__ gemm_out,
    int out_dim,
    int population_size,
    uint32_t seed,
    int shift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = population_size * out_dim;
    if (idx >= total) return;
    
    int pop = idx / out_dim;
    int r = idx % out_dim;
    int pair = pop / 2;
    int is_minus = pop & 1;
    uint32_t pair_seed = seed + pair;
    
    int32_t val = gemm_out[idx] >> shift;
    int8_t noise = perturb_gen_noise(pair_seed, r);
    val += is_minus ? -(int32_t)noise : (int32_t)noise;
    
    out_i8[idx] = (int8_t)max(-127, min(127, val));
}

inline void print_int8_tc_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("INT8 Tensor Core Info:\n");
    printf("  GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Mode: cuBLASLt INT8\n");
}

#endif
