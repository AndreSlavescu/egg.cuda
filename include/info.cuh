#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

namespace egg {

inline size_t compute_total_params(int vocab_size, int hidden_dim, int n_layers, int mlp_expand_dim) {
    size_t emb_params = vocab_size * hidden_dim;
    size_t gru_params = n_layers * 4 * hidden_dim * hidden_dim;
    size_t gru_bias_params = n_layers * 2 * hidden_dim;
    size_t mlp_params = n_layers * 2 * hidden_dim * mlp_expand_dim;
    size_t head_params = hidden_dim * vocab_size;
    size_t ln_params = n_layers * 2 * hidden_dim + hidden_dim;
    return emb_params + gru_params + gru_bias_params + mlp_params + head_params + ln_params;
}

inline void print_config(int population_size, int hidden_dim, int mlp_expand_dim, 
                         int n_layers, int seq_len, int vocab_size, bool is_int8_tc) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    size_t total_params = compute_total_params(vocab_size, hidden_dim, n_layers, mlp_expand_dim);
    
    printf("\n================ MODEL CONFIGURATION ================\n");
    printf("  Device:          %s\n", prop.name);
    printf("  Population Size: %d\n", population_size);
    printf("  Hidden Dim:      %d\n", hidden_dim);
    printf("  MLP Expand Dim:  %d\n", mlp_expand_dim);
    printf("  Num Layers:      %d\n", n_layers);
    printf("  Seq Len:         %d\n", seq_len);
    printf("  Vocab Size:      %d\n", vocab_size);
    printf("  Total Params:    %.2f M\n", total_params / 1e6);
    printf("  Precision:       %s\n", is_int8_tc ? "INT8 Training" : "FP32 Training");
    printf("=======================================================\n\n");
    fflush(stdout);
}

inline void print_config_distributed(int num_gpus, int pop_per_gpu, int total_pop,
                                     int hidden_dim, int mlp_expand_dim,
                                     int n_layers, int seq_len, int vocab_size, bool is_int8_tc) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    size_t total_params = compute_total_params(vocab_size, hidden_dim, n_layers, mlp_expand_dim);
    
    printf("\n================ MODEL CONFIGURATION ================\n");
    printf("  Device:          %s\n", prop.name);
    printf("  Num GPUs:        %d\n", num_gpus);
    printf("  Population/GPU:  %d\n", pop_per_gpu);
    printf("  Total Population:%d\n", total_pop);
    printf("  Hidden Dim:      %d\n", hidden_dim);
    printf("  MLP Expand Dim:  %d\n", mlp_expand_dim);
    printf("  Num Layers:      %d\n", n_layers);
    printf("  Seq Len:         %d\n", seq_len);
    printf("  Vocab Size:      %d\n", vocab_size);
    printf("  Total Params:    %.2f M\n", total_params / 1e6);
    printf("  Precision:       %s\n", is_int8_tc ? "INT8 Training" : "FP32 Training");
    printf("=======================================================\n\n");
    fflush(stdout);
}

inline void print_gpu_topology() {
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    printf("=== GPU Cluster Topology ===\n");
    for (int i = 0; i < num_gpus; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU %d: %s (SM %.1f, %d SMs, %.1f GB)\n",
            i, prop.name, (float)prop.major + (float)prop.minor / 10.0f,
            prop.multiProcessorCount, (float)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        
        printf("  P2P peers: ");
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) continue;
            int can_access;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            if (can_access) printf("%d ", j);
        }
        printf("\n");
    }
    printf("============================\n\n");
}

inline void print_training_start(int hidden_dim, int n_layers, int seq_len, 
                                  int pop_per_gpu, int num_gpus, int total_pop, bool is_int8_tc) {
    printf("Training: H=%d L=%d S=%d P=%d x %d GPUs = %d total (%s)\n",
           hidden_dim, n_layers, seq_len, pop_per_gpu, num_gpus, total_pop,
           is_int8_tc ? "INT8 Synchronized" : "TF32 Synchronized");
    fflush(stdout);
}

inline void print_distributed_step(long step, float avg_loss, int threshold, 
                                    int pos, int neg, float ms,
                                    int num_gpus, float sync_mb) {
    printf("Step %ld | Loss: %.4f | Thr: %d | +: %d | -: %d | %.1f ms\n",
           step, avg_loss, threshold, pos, neg, ms);
    printf("reduce: GPU[0-%d] -> host | update: GPU[0-%d] | sync: GPU0 -> [1-%d] (%.1fMB via NVLink)\n\n",
           num_gpus-1, num_gpus-1, num_gpus-1, sync_mb);
    fflush(stdout);
}

inline void print_int8_tc_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("INT8 Tensor Core: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
}

} // namespace egg
