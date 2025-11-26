#ifndef INT8_TC_CUH
#define INT8_TC_CUH

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <stdio.h>
#include <unordered_map>

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

__global__ void quantize_kernel(int8_t* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = fminf(fmaxf(in[i], -127.0f), 127.0f);
        out[i] = (int8_t)__float2int_rn(val);
    }
}

__global__ void dequantize_kernel(float* out, const int32_t* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
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
