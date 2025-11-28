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

__global__ void __launch_bounds__(256)
quantize_dynamic_kernel(int8_t* __restrict__ out, float* __restrict__ scales, 
                        const float* __restrict__ in, int dim, int batch) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float s_absmax;
    __shared__ float warp_max[8];
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    float local_max = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(in[b * dim + i]));
    }
    
    float warp_m = cg::reduce(warp, local_max, cg::greater<float>());
    if (warp.thread_rank() == 0) warp_max[tid / 32] = warp_m;
    block.sync();
    
    if (tid == 0) {
        float m = 0.0f;
        for (int i = 0; i < 8; i++) m = fmaxf(m, warp_max[i]);
        s_absmax = fmaxf(m, 1e-6f);
        scales[b] = s_absmax / 127.0f;
    }
    block.sync();
    
    float scale = 127.0f / s_absmax;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = in[b * dim + i] * scale;
        out[b * dim + i] = (int8_t)__float2int_rn(fminf(fmaxf(v, -127.0f), 127.0f));
    }
}

__global__ void __launch_bounds__(256)
dequant_dynamic_kernel(float* __restrict__ out, const int32_t* __restrict__ in,
                       const float* __restrict__ scales_x, float scale_w,
                       int dim, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * dim) return;
    
    int b = idx / dim;
    float scale = scales_x[b] * scale_w;
    out[idx] = (float)in[idx] * scale;
}

#endif
