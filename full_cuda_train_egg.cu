#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <signal.h>
#include <unistd.h>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
    const char msg[] = "\n[SIGINT] Interrupt received. Stopping after current step...\n";
    write(STDOUT_FILENO, msg, sizeof(msg)-1);
    keep_running = 0;
}

// --- Configuration ---
#define SM_CORES 128
#define WARP_SIZE 32
#define BLOCK_THREADS 256
#define BATCH WARP_SIZE
#define VOCAB_SIZE 256
#define HIDDEN_DIM (SM_CORES * 1) 
#define N_LAYERS 4
#define SEQ_LEN 256
#define POPULATION_SIZE (SM_CORES * 512)
#define SHARED_STRIDE (HIDDEN_DIM * 4)
#define FIXED_POINT 4
#define SIGMA_SHIFT 4
#define SIGMA_SHIFT_VECTOR (SIGMA_SHIFT - 2)
#define MAX_VAL 127
#define MIN_VAL -127
#define MAX_STRIDE 8 

// --- Seed Offsets ---
#define SEED_OFF_EMB 0
#define SEED_OFF_GRU_M1 1
#define SEED_OFF_GRU_M2 2
#define SEED_OFF_GRU_M3 3
#define SEED_OFF_GRU_M4 4
#define SEED_OFF_MLP_EXP 5
#define SEED_OFF_MLP_PROJ 6
#define SEED_OFF_GRU_B1 7
#define SEED_OFF_GRU_B2 8
#define SEED_OFF_LN_W1 9
#define SEED_OFF_LN_W2 10
#define SEED_OFF_LN_OUT 11
#define SEED_OFF_HEAD 999

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// --- Data Structures ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM]; 
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM]; 
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; 
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; 
    int8_t head[HIDDEN_DIM * VOCAB_SIZE]; 
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM];
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// Global Tables
__constant__ int32_t d_EXP2_TABLE[256];
int32_t h_EXP2_TABLE[256];

__device__ int32_t d_debug_updates[2]; // 0: Inc, 1: Dec

// --- Helpers (Host) ---

int get_update_threshold(double loss) {
    if (loss > 5.0) return 1000;
    if (loss > 4.0) return 5000;
    if (loss > 3.8) return 30000;
    if (loss > 3.6) return 60000;
    if (loss > 3.4) return 90000;
    if (loss > 3.2) return 120000;
    if (loss > 3.0) return 150000;
    if (loss > 1.0) return 270000;
    return 400000; 
}

double get_time_diff_ms(struct timespec start, struct timespec end) {
    return ((end.tv_sec - start.tv_sec) * 1000.0) + 
           ((end.tv_nsec - start.tv_nsec) / 1e6);
}

void init_tables() {
    for(int i=0; i<256; i++) 
        h_EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

static inline uint32_t xorshift32_host(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_host(uint32_t *rng) {
    uint32_t r = xorshift32_host(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

void transpose_matrix(int8_t *dst, int8_t *src, int rows, int cols) {
    for(int r=0; r<rows; r++) {
        for(int c=0; c<cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void init_model(EggModel *model) {
    uint32_t rng = 42;
    EggModel *temp = (EggModel*)malloc(sizeof(EggModel)); 
    if (!temp) { printf("Failed to allocate temp model\n"); exit(1); }

    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->embedding[i] = gen_noise_host(&rng);
    transpose_matrix(model->embedding, temp->embedding, VOCAB_SIZE, HIDDEN_DIM);

    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) temp->head[i] = gen_noise_host(&rng);
    transpose_matrix(model->head, temp->head, VOCAB_SIZE, HIDDEN_DIM);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) temp->gru_weights[0][0][i] = gen_noise_host(&rng);
            transpose_matrix(model->gru_weights[l][g], temp->gru_weights[0][0], HIDDEN_DIM, HIDDEN_DIM);
        }
        for(int m=0; m<2; m++) for(int i=0; i<HIDDEN_DIM; i++) model->gru_biases[l][m][i] = 0; 
    }
    
    for(int l=0; l<N_LAYERS; l++) {
        int dim_expand = HIDDEN_DIM * (HIDDEN_DIM * 4);
        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][0], temp->mlp_weights[0][0], HIDDEN_DIM*4, HIDDEN_DIM);

        for(int i=0; i<dim_expand; i++) temp->mlp_weights[0][0][i] = gen_noise_host(&rng);
        transpose_matrix(model->mlp_weights[l][1], temp->mlp_weights[0][0], HIDDEN_DIM, HIDDEN_DIM*4);
        
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
    }
    
    for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    
    free(temp);
}


// --- DEVICE ---
#define KERNEL_LOOP(idx, limit) for(int idx = threadIdx.x; idx < (limit); idx += blockDim.x)

__device__ __forceinline__ uint32_t hash_rng(uint32_t s, uint32_t idx) {
    uint32_t x = s + idx * 0x9e3779b9u; 
    x ^= x >> 16; x *= 0x85ebca6b; x ^= x >> 13; x *= 0xc2b2ae35; x ^= x >> 16;
    return x;
}

__device__ __forceinline__ int8_t noise_from_hash(uint32_t s, uint32_t idx) {
    uint32_t r = hash_rng(s, idx);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

__device__ __forceinline__ int8_t clip(long long a) {
    return (a > MAX_VAL) ? MAX_VAL : ((a < MIN_VAL) ? MIN_VAL : (int8_t)a);
}

// Helper to broadcast 64-bit value from a lane to all threads in warp
__device__ __forceinline__ long long warpBroadcast(long long val, int src_lane) {
    int lo = __shfl_sync(0xFFFFFFFF, (int)val, src_lane);
    int hi = __shfl_sync(0xFFFFFFFF, (int)(val >> 32), src_lane);
    return ((long long)hi << 32) | (unsigned int)lo;
}

extern __shared__ int8_t s_mem[]; 

__global__ void generate_sequence_kernel(
    const EggModel * __restrict__ model,
    uint32_t seed,
    const uint8_t *seed_text,
    int seed_len,
    int gen_len,
    uint8_t *output
) {
    int8_t *s_ptr = s_mem;
    
    // CUB definitions
    typedef cub::BlockReduce<long long, BLOCK_THREADS> BlockReduce;
    typedef cub::BlockScan<long long, BLOCK_THREADS> BlockScan;
    
    // Shared storage for CUB. Using a union to save memory.
    __shared__ union {
        typename BlockReduce::TempStorage reduce;
        typename BlockScan::TempStorage scan;
    } temp_storage;

    __shared__ long long shared_sum; // For broadcasting reduction result

    // Local hidden state
    int8_t h_local[N_LAYERS][MAX_STRIDE];
    KERNEL_LOOP(i, HIDDEN_DIM) {
        for(int l=0; l<N_LAYERS; l++) h_local[l][(i - threadIdx.x)/blockDim.x] = 0;
    }

    int total_len = seed_len + gen_len;
    uint8_t current_token = 0; 
    
    for(int t=0; t < total_len; t++) {
        if (t < seed_len) current_token = seed_text[t];
        
        // 1. Embed
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = model->embedding[i * VOCAB_SIZE + current_token];
        __syncthreads();
        
        // 2. Layers
        for(int l=0; l<N_LAYERS; l++) {
             // LN 1
             long long local_sum = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
             
             long long sum = BlockReduce(temp_storage.reduce).Sum(local_sum);
             // Broadcast result to all threads
             if (threadIdx.x == 0) shared_sum = sum;
             __syncthreads();
             sum = shared_sum;

             if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_weights[l][0][i]) / mean);
             __syncthreads();
             
             // Copy H to Shared (Buf 1)
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[HIDDEN_DIM + i] = h_local[l][(i - threadIdx.x)/blockDim.x];
             __syncthreads();
             
             // GRU Phase 1
             const int8_t *w0 = &model->gru_weights[l][0][0];
             const int8_t *w1 = &model->gru_weights[l][1][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc0 = 0, acc1 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc0 += (long long)s_ptr[k] * w0[k*HIDDEN_DIM + i];
                     acc1 += (long long)s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + i];
                 }
                 s_ptr[3*HIDDEN_DIM + i] = acc0 >> 8;
                 s_ptr[2*HIDDEN_DIM + i] = acc1 >> 8;
             }
             __syncthreads();
             
             // Gates
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 int8_t b1 = s_ptr[3*HIDDEN_DIM + i];
                 int8_t b2 = s_ptr[2*HIDDEN_DIM + i];
                 int8_t ft = clip((long long)b1 + b2 + model->gru_biases[l][0][i]);
                 
                 int8_t h_val = s_ptr[HIDDEN_DIM + i];
                 int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                 
                 s_ptr[3*HIDDEN_DIM + i] = ft;
                 s_ptr[2*HIDDEN_DIM + i] = gated;
             }
             __syncthreads();
             
             // GRU Phase 2
             const int8_t *w2 = &model->gru_weights[l][2][0];
             const int8_t *w3 = &model->gru_weights[l][3][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc0 = 0, acc1 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                     acc0 += (long long)s_ptr[k] * w2[k*HIDDEN_DIM + i];
                     acc1 += (long long)s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + i];
                 }
                 int8_t ft = s_ptr[3*HIDDEN_DIM + i];
                 int8_t ht = clip((acc0 >> 8) + (acc1 >> 8) + model->gru_biases[l][1][i]);
                 
                 int8_t h_curr = h_local[l][(i - threadIdx.x)/blockDim.x];
                 int32_t diff = ht - h_curr;
                 int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                 h_curr = clip(h_curr + update);
                 h_local[l][(i - threadIdx.x)/blockDim.x] = h_curr;
                 
                 s_ptr[i] = clip((long long)h_curr + s_ptr[i]);
             }
             __syncthreads();
             
             // Save MLP Residual
             int8_t mlp_resid[MAX_STRIDE];
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 mlp_resid[(i - threadIdx.x)/blockDim.x] = s_ptr[i];
             }
             
             // MLP LN
             local_sum = 0;
             KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
             sum = BlockReduce(temp_storage.reduce).Sum(local_sum);
             if (threadIdx.x == 0) shared_sum = sum;
             __syncthreads();
             sum = shared_sum;

             if(!sum) sum=1; mean = sum/HIDDEN_DIM; if(!mean) mean=1;
             
             KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_weights[l][1][i]) / mean);
             __syncthreads();
             
             // MLP Expand
             int8_t exp_res[MAX_STRIDE][4];
             const int8_t *w_exp = &model->mlp_weights[l][0][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                     long long acc = 0;
                     for(int k=0; k<HIDDEN_DIM; k++) {
                         acc += (long long)s_ptr[k] * w_exp[k*(4*HIDDEN_DIM) + (i + sub*HIDDEN_DIM)];
                     }
                     exp_res[(i - threadIdx.x)/blockDim.x][sub] = clip(acc >> 8);
                 }
             }
             __syncthreads();
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 for(int sub=0; sub<4; sub++) {
                     s_ptr[i + sub*HIDDEN_DIM] = exp_res[(i - threadIdx.x)/blockDim.x][sub];
                 }
             }
             __syncthreads();
             
             // MLP Project
             const int8_t *w_proj = &model->mlp_weights[l][1][0];
             
             KERNEL_LOOP(i, HIDDEN_DIM) {
                 long long acc = 0;
                 for(int k=0; k<(4*HIDDEN_DIM); k++) {
                     acc += (long long)s_ptr[k] * w_proj[k*HIDDEN_DIM + i];
                 }
                 s_ptr[i] = clip((acc >> 9) + mlp_resid[(i - threadIdx.x)/blockDim.x]);
             }
             __syncthreads();
        }
        
        // 3. Head
        long long local_sum = 0;
        KERNEL_LOOP(i, HIDDEN_DIM) local_sum += abs((long long)s_ptr[i]);
        long long sum = BlockReduce(temp_storage.reduce).Sum(local_sum);
        if (threadIdx.x == 0) shared_sum = sum;
        __syncthreads();
        sum = shared_sum;        

        if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        
        KERNEL_LOOP(i, HIDDEN_DIM) s_ptr[i] = clip(((long long)s_ptr[i] * model->ln_out[i]) / mean);
        __syncthreads();
        
        // Logits
        const int8_t *w_head = &model->head[0];
        KERNEL_LOOP(i, VOCAB_SIZE) {
            long long acc = 0;
            for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (long long)s_ptr[k] * w_head[k*VOCAB_SIZE + i]; 
            }
            s_ptr[HIDDEN_DIM + i] = clip(acc >> 8); 
        }
        __syncthreads();
        
        // 4. Sample
        if (t >= seed_len) {
            long long val = 0;
            KERNEL_LOOP(i, VOCAB_SIZE) {
                int idx = (int32_t)s_ptr[HIDDEN_DIM + i] + 128;
                idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
                val = d_EXP2_TABLE[idx];
            }
            // Block Reduce to get Sum
            long long sum_exp = BlockReduce(temp_storage.reduce).Sum(val);
            if (threadIdx.x == 0) shared_sum = sum_exp;
            __syncthreads();
            sum_exp = shared_sum; 
            
            uint32_t s = seed + t * 222;
            uint32_t r_val = hash_rng(s, 0);
            long long r_threshold = (sum_exp > 0) ? (r_val % sum_exp) : 0;
            
            // Parallel Selection using BlockScan
            long long prefix_sum;
            long long block_aggregate; 
            BlockScan(temp_storage.scan).InclusiveSum(val, prefix_sum, block_aggregate);
            
            __shared__ int selected_token;
            if (threadIdx.x == 0) selected_token = VOCAB_SIZE - 1; // Default
            __syncthreads();
            
            if (prefix_sum > r_threshold) {
                long long prev_sum = prefix_sum - val;
                if (prev_sum <= r_threshold) {
                    // Logic implies this thread holds the target range
                    // Note: If VOCAB_SIZE > BLOCK_THREADS, we need mapping.
                    // Assuming VOCAB_SIZE (256) <= BLOCK_THREADS (256) as per config
                    atomicMin(&selected_token, threadIdx.x);
                }
            }
            __syncthreads();
            
            current_token = (uint8_t)selected_token;
            output[t - seed_len] = current_token;
        }
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS) train_sequence_kernel(
    const uint8_t * __restrict__ dataset,
    long data_len,
    int start_idx,
    const EggModel * __restrict__ model,
    int8_t * __restrict__ pop_states, 
    int32_t *accum_loss,
    uint32_t step_seed
) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int p_idx = blockIdx.x * (BLOCK_THREADS / WARP_SIZE) + warp_id;
    
    if (p_idx >= POPULATION_SIZE) return;

    int8_t *my_s_ptr = &s_mem[warp_id * SHARED_STRIDE];

    // CUB WarpReduce
    typedef cub::WarpReduce<long long> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_THREADS / WARP_SIZE];
    
    // Load State
    int8_t h_local[N_LAYERS][MAX_STRIDE];
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE;
        for(int l=0; l<N_LAYERS; l++) {
             h_local[l][sub] = pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i];
        }
    }
    
    long long my_loss = 0;
    long pair_idx = p_idx / 2;
    long stride = data_len / (POPULATION_SIZE / 2);
    long stream_pos = (start_idx + (pair_idx * stride)) % (data_len - SEQ_LEN);
    int ns = (p_idx % 2 == 0) ? 1 : -1;

    for (int t = 0; t < SEQ_LEN; t++) {
        __syncwarp();
        
        uint8_t input_token = dataset[stream_pos + t];
        uint32_t seed_emb = (step_seed + pair_idx) + SEED_OFF_EMB;
        int8_t a_token = noise_from_hash(seed_emb, input_token);
        
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            int8_t base = model->embedding[i * VOCAB_SIZE + input_token];
            int8_t b_val = noise_from_hash(seed_emb + HIDDEN_DIM, i);
            long long perturb = ((long long)a_token * b_val * ns) >> (FIXED_POINT + SIGMA_SHIFT);
            my_s_ptr[i] = clip((long long)base + perturb);
        }
        __syncwarp();
        
        for (int l = 0; l < N_LAYERS; l++) {
            uint32_t seed = (step_seed + pair_idx) + (l * 100);

            // LN 1
            long long local_sum = 0;
            int8_t gru_resid[MAX_STRIDE];

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t val = my_s_ptr[i];
                gru_resid[i / WARP_SIZE] = val;
                local_sum += abs((long long)val);
            }
            long long sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
            sum = warpBroadcast(sum, 0); // Broadcast to all lanes

            if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t w = model->ln_weights[l][0][i];
                int8_t a = noise_from_hash(seed + SEED_OFF_LN_W1, i);
                long long perturb = ((long long)a * ns) >> SIGMA_SHIFT_VECTOR;
                my_s_ptr[i] = clip(((long long)my_s_ptr[i] * (w + perturb)) / mean);
            }

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 my_s_ptr[HIDDEN_DIM + i] = h_local[l][i / WARP_SIZE];
            }
            __syncwarp();

            // Rank-1 Precalc
            long long ls_1 = 0, ls_2 = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b1 = noise_from_hash(seed + SEED_OFF_GRU_M1 + HIDDEN_DIM, i);
                ls_1 += (long long)my_s_ptr[i] * b1;
                
                int8_t b2 = noise_from_hash(seed + SEED_OFF_GRU_M2 + HIDDEN_DIM, i);
                ls_2 += (long long)my_s_ptr[HIDDEN_DIM + i] * b2;
            }
            long long xB_m1 = WarpReduce(temp_storage[warp_id]).Sum(ls_1);
            xB_m1 = warpBroadcast(xB_m1, 0);
            
            long long xB_m2 = WarpReduce(temp_storage[warp_id]).Sum(ls_2);
            xB_m2 = warpBroadcast(xB_m2, 0);

            // MatMul 1 & 2 + Gates
            const int8_t *w0 = &model->gru_weights[l][0][0];
            const int8_t *w1 = &model->gru_weights[l][1][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 long long dot1 = 0, dot2 = 0;
                 for(int k=0; k<HIDDEN_DIM; k++) {
                    dot1 += (long long)my_s_ptr[k] * w0[k*HIDDEN_DIM + i];
                    dot2 += (long long)my_s_ptr[HIDDEN_DIM + k] * w1[k*HIDDEN_DIM + i];
                 }
                 
                 int8_t a1 = noise_from_hash(seed + SEED_OFF_GRU_M1, i);
                 if(ns!=0) dot1 += ((xB_m1 * (long long)a1) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t a2 = noise_from_hash(seed + SEED_OFF_GRU_M2, i);
                 if(ns!=0) dot2 += ((xB_m2 * (long long)a2) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int8_t b_gate = model->gru_biases[l][0][i];
                 int8_t a_gate = noise_from_hash(seed + SEED_OFF_GRU_B1, i);
                 long long p_gate = ((long long)a_gate * ns) >> SIGMA_SHIFT_VECTOR;

                 int8_t ft = clip((dot1 >> 8) + (dot2 >> 8) + (b_gate + p_gate));
                 int8_t h_val = my_s_ptr[HIDDEN_DIM + i];
                 int8_t gated = (int8_t)(((long long)(ft + 127) * h_val) >> 8);
                 
                 my_s_ptr[2*HIDDEN_DIM + i] = gated; 
                 my_s_ptr[3*HIDDEN_DIM + i] = ft;    
            }
            __syncwarp();
            
            // Rank-1 M3 & M4
            long long ls_3=0, ls_4=0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b3 = noise_from_hash(seed + SEED_OFF_GRU_M3 + HIDDEN_DIM, i);
                ls_3 += (long long)my_s_ptr[i] * b3;
                
                int8_t b4 = noise_from_hash(seed + SEED_OFF_GRU_M4 + HIDDEN_DIM, i);
                ls_4 += (long long)my_s_ptr[2*HIDDEN_DIM + i] * b4;
            }
            long long xB_m3 = WarpReduce(temp_storage[warp_id]).Sum(ls_3);
            xB_m3 = warpBroadcast(xB_m3, 0);

            long long xB_m4 = WarpReduce(temp_storage[warp_id]).Sum(ls_4);
            xB_m4 = warpBroadcast(xB_m4, 0);
            
            // Phase 2
            const int8_t *w2 = &model->gru_weights[l][2][0];
            const int8_t *w3 = &model->gru_weights[l][3][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                long long dot1 = 0, dot2 = 0;
                for(int k=0; k<HIDDEN_DIM; k++) {
                    dot1 += (long long)my_s_ptr[k] * w2[k*HIDDEN_DIM + i];
                    dot2 += (long long)my_s_ptr[2*HIDDEN_DIM + k] * w3[k*HIDDEN_DIM + i];
                }
                
                int8_t a3 = noise_from_hash(seed + SEED_OFF_GRU_M3, i);
                if(ns!=0) dot1 += ((xB_m3 * (long long)a3) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                
                int8_t a4 = noise_from_hash(seed + SEED_OFF_GRU_M4, i);
                if(ns!=0) dot2 += ((xB_m4 * (long long)a4) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                
                int8_t b_ht = model->gru_biases[l][1][i];
                int8_t a_ht = noise_from_hash(seed + SEED_OFF_GRU_B2, i);
                long long p_ht = ((long long)a_ht * ns) >> SIGMA_SHIFT_VECTOR;

                int8_t ht = clip((dot1 >> 8) + (dot2 >> 8) + (b_ht + p_ht));
                
                int8_t ft = my_s_ptr[3*HIDDEN_DIM + i];
                int8_t h_curr = h_local[l][i / WARP_SIZE];
                int32_t diff = ht - h_curr;
                int32_t update = ((int32_t)(ft + 127) * diff) >> 8;
                h_curr = clip(h_curr + update);
                h_local[l][i / WARP_SIZE] = h_curr;
                
                my_s_ptr[i] = clip((long long)h_curr + gru_resid[i / WARP_SIZE]); 
            }
            __syncwarp(); 
            
            // Pre-LN Resid
            int8_t mlp_resid[MAX_STRIDE];
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                mlp_resid[i / WARP_SIZE] = my_s_ptr[i];
            }

            // MLP LN 2
            long long local_mlp_sum = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                local_mlp_sum += abs((long long)my_s_ptr[i]);
            }
            long long sum_mlp = WarpReduce(temp_storage[warp_id]).Sum(local_mlp_sum);
            sum_mlp = warpBroadcast(sum_mlp, 0);

            if(!sum_mlp) sum_mlp=1; long long mean_mlp = sum_mlp/HIDDEN_DIM; if(!mean_mlp) mean_mlp=1;

            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t w = model->ln_weights[l][1][i];
                int8_t a = noise_from_hash(seed + SEED_OFF_LN_W2, i);
                long long p = ((long long)a * ns) >> SIGMA_SHIFT_VECTOR;
                my_s_ptr[i] = clip(((long long)my_s_ptr[i] * (w + p)) / mean_mlp);
            }
            __syncwarp(); 
            
            // Expand
            int8_t mlp_res[MAX_STRIDE][4];
            
            uint32_t seed_exp = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_EXP;
            long long ls_exp = 0;
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                int8_t b_val = noise_from_hash(seed_exp + SHARED_STRIDE, i); 
                ls_exp += (long long)my_s_ptr[i] * b_val;
            }
            long long xB_mlp1 = WarpReduce(temp_storage[warp_id]).Sum(ls_exp);
            xB_mlp1 = warpBroadcast(xB_mlp1, 0);

            const int8_t *w_mlp1 = &model->mlp_weights[l][0][0];
            
            for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                      int out_idx = i + sub * HIDDEN_DIM;
                      long long acc_mlp = 0;
                      
                      for(int k=0; k<HIDDEN_DIM; k++) {
                          acc_mlp += (long long)my_s_ptr[k] * w_mlp1[k*(4*HIDDEN_DIM) + out_idx];
                      }
                      
                      int8_t a_val = noise_from_hash(seed_exp, out_idx);
                      if(ns!=0) acc_mlp += ((xB_mlp1 * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                      mlp_res[i / WARP_SIZE][sub] = clip(acc_mlp >> 8);
                 }
             }
             
             // Write Expand to Shared
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                     my_s_ptr[i + sub*HIDDEN_DIM] = mlp_res[i / WARP_SIZE][sub];
                 }
             }
             __syncwarp();
             
             // xB for M2
             uint32_t seed_proj = (step_seed+pair_idx) + (l * 100) + SEED_OFF_MLP_PROJ;
             long long ls_proj = 0;
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 for(int sub=0; sub<4; sub++) {
                     int in_idx = i + sub*HIDDEN_DIM;
                     int8_t b_val = noise_from_hash(seed_proj + HIDDEN_DIM, in_idx);
                     ls_proj += (long long)my_s_ptr[in_idx] * b_val;
                 }
             }
             long long xB_mlp2 = WarpReduce(temp_storage[warp_id]).Sum(ls_proj);
             xB_mlp2 = warpBroadcast(xB_mlp2, 0);
             
             const int8_t *w_mlp2 = &model->mlp_weights[l][1][0];
             for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
                 long long acc_proj = 0;
                 for(int k=0; k<HIDDEN_DIM*4; k++) {
                     acc_proj += (long long)my_s_ptr[k] * w_mlp2[k*HIDDEN_DIM + i];
                 }
                 int8_t a_val = noise_from_hash(seed_proj, i);
                 if(ns!=0) acc_proj += ((xB_mlp2 * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
                 
                 int32_t res = acc_proj >> 8; 
                 my_s_ptr[i] = clip(res + mlp_resid[i / WARP_SIZE]);
             }
             __syncwarp();
        } 
        
        // 3. Head LN
        long long local_sum = 0;
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
            local_sum += abs((long long)my_s_ptr[i]);
        }
        long long sum = WarpReduce(temp_storage[warp_id]).Sum(local_sum);
        sum = warpBroadcast(sum, 0);
        if(!sum) sum=1; long long mean = sum/HIDDEN_DIM; if(!mean) mean=1;
        
        uint32_t seed_ln_out = (step_seed+pair_idx);
        for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
             int8_t w = model->ln_out[i];
             int8_t a = noise_from_hash(seed_ln_out + SEED_OFF_LN_OUT, i);
             long long p = ((long long)a * ns) >> SIGMA_SHIFT_VECTOR;
             my_s_ptr[i] = clip(((long long)my_s_ptr[i] * (w + p)) / mean);
        }
        __syncwarp();
        
        // 4. Head Dense & Loss
         uint32_t seed_head = (step_seed+pair_idx) + SEED_OFF_HEAD;
         long long ls_head = 0;
         for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
             int8_t b_val = noise_from_hash(seed_head + VOCAB_SIZE, i); 
             ls_head += (long long)my_s_ptr[i] * b_val;
         }
         long long xB_head = WarpReduce(temp_storage[warp_id]).Sum(ls_head);
         xB_head = warpBroadcast(xB_head, 0);
         
         const int8_t *w_head = &model->head[0];
         
         for(int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
             long long acc = 0;
             for(int k=0; k<HIDDEN_DIM; k++) {
                acc += (long long)my_s_ptr[k] * w_head[k*VOCAB_SIZE + i];
             }
            
             int8_t a_val = noise_from_hash(seed_head, i);
             if(ns!=0) acc += ((xB_head * (long long)a_val) * ns) >> (FIXED_POINT + SIGMA_SHIFT);
             
             my_s_ptr[HIDDEN_DIM + i] = clip(acc >> 8); 
         }
         __syncwarp();
         
          // 5. Softmax Loss
          uint8_t target_token = dataset[stream_pos + t + 1];
          
          long long local_exp = 0;
          for(int i = lane_id; i < VOCAB_SIZE; i += WARP_SIZE) {
              int idx = (int32_t)my_s_ptr[HIDDEN_DIM + i] + 128;
              idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
              local_exp += d_EXP2_TABLE[idx];
          }
          long long sum_exp = WarpReduce(temp_storage[warp_id]).Sum(local_exp);
          sum_exp = warpBroadcast(sum_exp, 0);
          
          if(lane_id == 0) {
             long long log_sum = 0;
             long long x = sum_exp;
             if (x > 0) {
                 int pos = 0;
                 while(x >= 65536) { x >>= 16; pos += 16; }
                 if (x >= 256)  { x >>= 8;  pos += 8; }
                 if (x >= 16)   { x >>= 4;  pos += 4; }
                 if (x >= 4)    { x >>= 2;  pos += 2; }
                 if (x >= 2)    {           pos += 1; }
                 
                 long long fraction = (pos>=4) ? (sum_exp-(1LL<<pos))>>(pos-4) : (sum_exp-(1LL<<pos))<<(4-pos);
                 log_sum = (pos<<4) + fraction - 64;
             }
             
             int32_t target_l = (int32_t)my_s_ptr[HIDDEN_DIM + target_token] + 128;
             my_loss += (log_sum - target_l);
          }
    } 
    
    if (lane_id == 0) {
        accum_loss[p_idx] = (int32_t)my_loss;
    }
    
    // Store State
    for (int i = lane_id; i < HIDDEN_DIM; i += WARP_SIZE) {
        int sub = i / WARP_SIZE; 
        for(int l=0; l<N_LAYERS; l++) {
             pop_states[p_idx * (N_LAYERS * HIDDEN_DIM) + l * HIDDEN_DIM + i] = h_local[l][sub];
        }
    }
}

__global__ void compute_fitness_kernel(
    const int32_t *__restrict__ accum_loss, 
    int32_t *__restrict__ fitnesses, 
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int32_t loss_pos = accum_loss[2*idx];
    int32_t loss_neg = accum_loss[2*idx+1];
    
    int32_t fit = 0;
    if (loss_pos < loss_neg) fit = 1;
    else if (loss_neg < loss_pos) fit = -1;
    
    fitnesses[idx] = fit;
}

__global__ void update_matrix_kernel(
    int8_t * __restrict__ W,
    int rows,
    int cols,
    int offset_A,
    int offset_B,
    int seed_base_add,
    const int32_t * __restrict__ fitnesses,
    uint32_t step_seed,
    int threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    
    int r = idx % rows; 
    int c = idx / rows; 
    
    long long vote = 0;
    
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p];
        if (fit == 0) continue;

        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + offset_A, r);
        int8_t b = noise_from_hash(s + offset_B, c);
        
        vote += (long long)fit * (int)a * (int)b;
    }
    
    int8_t w_curr = W[c * rows + r]; 

    if(vote > threshold && w_curr < MAX_VAL) {
        w_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -threshold && w_curr > MIN_VAL) {
        w_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    W[c * rows + r] = w_curr; 
}

__global__ void update_vector_kernel(
    int8_t * __restrict__ V,
    int len,
    int seed_off_A,
    int seed_base_add,
    const int32_t * __restrict__ fitnesses,
    uint32_t step_seed,
    int threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    
    long long vote = 0;
    
    for(int p=0; p < POPULATION_SIZE/2; p++) {
        int fit = fitnesses[p];
        if (fit == 0) continue;
        
        uint32_t s = step_seed + p + seed_base_add;
        int8_t a = noise_from_hash(s + seed_off_A, idx);
        
        vote += (long long)fit * (int)a;
    }
    
    int8_t v_curr = V[idx];

    if(vote > threshold && v_curr < MAX_VAL) {
        v_curr++;
        atomicAdd(&d_debug_updates[0], 1);
    }
    else if(vote < -threshold && v_curr > MIN_VAL) {
        v_curr--;
        atomicAdd(&d_debug_updates[1], 1);
    }
    V[idx] = v_curr; 
}

struct abs_functor {
    __host__ __device__ int operator()(const int& x) const { return abs(x); }
};

int main() {
    signal(SIGINT, handle_sigint);
    srand(time(NULL));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    init_tables();
    cudaMemcpyToSymbol(d_EXP2_TABLE, h_EXP2_TABLE, 256*sizeof(int32_t));
    
    Dataset ds = {0,0};
    FILE *f = fopen("input.txt", "rb");
    if(!f) { printf("Error: input.txt not found!\n"); exit(1); }
    fseek(f,0,SEEK_END); ds.length=ftell(f); fseek(f,0,SEEK_SET); 
    ds.data=(uint8_t*)malloc(ds.length); 
    if(!fread(ds.data,1,ds.length,f)) { printf("Error reading input.txt\n"); exit(1); }
    fclose(f); 

    // --- Config Dump ---
    {
        long long params_embedding = (long long)VOCAB_SIZE * HIDDEN_DIM;
        long long params_gru = (long long)N_LAYERS * 4 * HIDDEN_DIM * HIDDEN_DIM;
        long long params_gru_bias = (long long)N_LAYERS * 2 * HIDDEN_DIM;
        long long params_mlp = (long long)N_LAYERS * 2 * 4 * HIDDEN_DIM * HIDDEN_DIM;
        long long params_ln = (long long)N_LAYERS * 2 * HIDDEN_DIM;
        long long params_ln_out = (long long)HIDDEN_DIM;
        long long params_head = (long long)HIDDEN_DIM * VOCAB_SIZE;
        long long total_params = params_embedding + params_gru + params_gru_bias + 
                                 params_mlp + params_ln + params_ln_out + params_head;

        printf("\n================ CONFIGURATION DUMP ================\n");
        printf("  Device: %s\n", prop.name);
        printf("  SM Cores:        %d\n", SM_CORES);
        printf("  Population Size: %d\n", POPULATION_SIZE);
        printf("  Hidden Dim:      %d\n", HIDDEN_DIM);
        printf("  Seq Len:         %d\n", SEQ_LEN);
        printf("  Total Params:    %.2f M\n", total_params/1000000.0);
        printf("====================================================\n\n");
    }

    EggModel *h_model = (EggModel*)malloc(sizeof(EggModel));
    init_model(h_model); 
    
    EggModel *d_model;
    CHECK_CUDA(cudaMalloc(&d_model, sizeof(EggModel)));
    CHECK_CUDA(cudaMemcpy(d_model, h_model, sizeof(EggModel), cudaMemcpyHostToDevice));
    
    uint8_t *d_dataset;
    CHECK_CUDA(cudaMalloc(&d_dataset, ds.length));
    CHECK_CUDA(cudaMemcpy(d_dataset, ds.data, ds.length, cudaMemcpyHostToDevice));
    
    int32_t *d_accum_loss;
    CHECK_CUDA(cudaMalloc(&d_accum_loss, POPULATION_SIZE * sizeof(int32_t)));
    
    int32_t *d_fitnesses;
    CHECK_CUDA(cudaMalloc(&d_fitnesses, (POPULATION_SIZE/2) * sizeof(int32_t)));
    
    int8_t *d_pop_states;
    size_t state_size = POPULATION_SIZE * N_LAYERS * HIDDEN_DIM;
    CHECK_CUDA(cudaMalloc(&d_pop_states, state_size));
    CHECK_CUDA(cudaMemset(d_pop_states, 0, state_size));
    
    printf("Starting EGGROLL CUDA Training (Batch=%d)...\n", BATCH);
    long max_steps = (ds.length - 1) / SEQ_LEN;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    unsigned long total_tokens = 0;

    for(long step=0; step<max_steps && keep_running; step++) {
        struct timespec t0, t1, t2, t3;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        uint32_t seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        int threads_per_block = BLOCK_THREADS;
        int warps_per_block = BLOCK_THREADS / WARP_SIZE;
        int blocks = POPULATION_SIZE / warps_per_block;
        size_t shared_mem_size = warps_per_block * SHARED_STRIDE;
        
        train_sequence_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_dataset, ds.length, start_idx, d_model, d_pop_states, d_accum_loss, seed
        );
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        
        // --- Host Control Removed: GPU logic via Thrust & Custom Kernels ---
        
        // 1. Compute Fitness on GPU
        int half_pop = POPULATION_SIZE / 2;
        compute_fitness_kernel<<< (half_pop + 255) / 256, 256 >>>(
            d_accum_loss, d_fitnesses, half_pop
        );
        
        // 2. Reduce Loss (Thrust)
        thrust::device_ptr<int32_t> t_accum_loss(d_accum_loss);
        long long current_accum_total = thrust::reduce(
            thrust::device, 
            t_accum_loss, 
            t_accum_loss + POPULATION_SIZE, 
            (long long)0, 
            thrust::plus<long long>()
        );
        
        double current_avg_loss = (double)current_accum_total / POPULATION_SIZE;
        double current_loss_per_token = current_avg_loss / (SEQ_LEN * (1 << FIXED_POINT));
        int current_threshold = get_update_threshold(current_loss_per_token);

        clock_gettime(CLOCK_MONOTONIC, &t2);
        
        if (!keep_running) break;

        // Debug prints
        if (step % 10 == 0) {
            thrust::device_ptr<int32_t> t_fitnesses(d_fitnesses);
            int fit_sum = thrust::transform_reduce(
                thrust::device,
                t_fitnesses, 
                t_fitnesses + half_pop,
                abs_functor(),
                0,
                thrust::plus<int>()
            );
            
            // Just peek at the first item's debug info by copying minimal data
            int32_t h_debug_loss[2];
            int32_t h_debug_fit[1];
            cudaMemcpy(h_debug_loss, d_accum_loss, 2*sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_debug_fit, d_fitnesses, sizeof(int32_t), cudaMemcpyDeviceToHost);

            printf("[Debug] Step %ld Pair 0: Pos=%d Neg=%d Fit=%d | Threshold=%d | Total NonZero Fits: %d\n", 
                step, h_debug_loss[0], h_debug_loss[1], h_debug_fit[0], current_threshold, fit_sum);
        }
        
        int32_t zeros[2] = {0, 0};
        CHECK_CUDA(cudaMemcpyToSymbol(d_debug_updates, zeros, 2*sizeof(int32_t)));

        for(int l=0; l<N_LAYERS; l++) {
             int seed_base = l * 100;
             size_t gru_size = HIDDEN_DIM * HIDDEN_DIM * sizeof(int8_t);
             long off_g = offsetof(EggModel, gru_weights) + (l * 4 * HIDDEN_DIM * HIDDEN_DIM);
             
             for(int g=0; g<4; g++) {
                 int seed_off = SEED_OFF_GRU_M1 + g; 
                 update_matrix_kernel<<< (HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                     (int8_t*)d_model + off_g + g*gru_size, 
                     HIDDEN_DIM, HIDDEN_DIM, 
                     seed_off, seed_off + HIDDEN_DIM, 
                     seed_base, d_fitnesses, seed, current_threshold
                 );
             }
             
             size_t mlp_exp_size = HIDDEN_DIM * (HIDDEN_DIM*4) * sizeof(int8_t);
             long off_m = offsetof(EggModel, mlp_weights) + (l * 2 * mlp_exp_size);
             
             update_matrix_kernel<<< (4*HIDDEN_DIM*HIDDEN_DIM + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + 0, 4*HIDDEN_DIM, HIDDEN_DIM, 
                 0, SHARED_STRIDE, 
                 seed_base + SEED_OFF_MLP_EXP, d_fitnesses, seed, current_threshold
             );
             
             update_matrix_kernel<<< (HIDDEN_DIM*(4*HIDDEN_DIM) + 511)/512, 512 >>>(
                 (int8_t*)d_model + off_m + mlp_exp_size, HIDDEN_DIM, 4*HIDDEN_DIM, 
                 0, HIDDEN_DIM, 
                 seed_base + SEED_OFF_MLP_PROJ, d_fitnesses, seed, current_threshold
             );

             long off_b = offsetof(EggModel, gru_biases) + (l * 2 * HIDDEN_DIM);
             update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_b, HIDDEN_DIM, SEED_OFF_GRU_B1, seed_base, d_fitnesses, seed, current_threshold);
             update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_b + HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_GRU_B2, seed_base, d_fitnesses, seed, current_threshold);

             long off_ln = offsetof(EggModel, ln_weights) + (l * 2 * HIDDEN_DIM);
             update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_ln, HIDDEN_DIM, SEED_OFF_LN_W1, seed_base, d_fitnesses, seed, current_threshold);
             update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_ln + HIDDEN_DIM, HIDDEN_DIM, SEED_OFF_LN_W2, seed_base, d_fitnesses, seed, current_threshold);
        }
        
        long off_ln_out = offsetof(EggModel, ln_out);
        update_vector_kernel<<< (HIDDEN_DIM+511)/512, 512 >>>((int8_t*)d_model + off_ln_out, HIDDEN_DIM, SEED_OFF_LN_OUT, 0, d_fitnesses, seed, current_threshold);

        long off_head = offsetof(EggModel, head);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_head, VOCAB_SIZE, HIDDEN_DIM, 
            0, VOCAB_SIZE, 
            SEED_OFF_HEAD, d_fitnesses, seed, current_threshold
        );
        
        long off_emb = offsetof(EggModel, embedding);
        update_matrix_kernel<<< (VOCAB_SIZE*HIDDEN_DIM + 511)/512, 512 >>>(
            (int8_t*)d_model + off_emb, HIDDEN_DIM, VOCAB_SIZE, 
            HIDDEN_DIM, 0,
            SEED_OFF_EMB, d_fitnesses, seed, current_threshold
        );
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t3);
        
        if (!keep_running) break;

        clock_gettime(CLOCK_MONOTONIC, &end);
        total_tokens += POPULATION_SIZE * SEQ_LEN;
        
        if (step % 1 == 0) { 
            int32_t h_updates[2];
            CHECK_CUDA(cudaMemcpyFromSymbol(h_updates, d_debug_updates, 2*sizeof(int32_t), 0, cudaMemcpyDeviceToHost));
            
            static uint8_t *d_output = NULL;
            static uint8_t *h_output = NULL;
            int seed_len = 30; 
            int gen_len = 50;
            if(!d_output) {
                CHECK_CUDA(cudaMalloc(&d_output, gen_len));
                h_output = (uint8_t*)malloc(gen_len);
            }
            
            generate_sequence_kernel<<<1, 256, 5 * HIDDEN_DIM * sizeof(int8_t)>>>(
                d_model, seed + 12345, d_dataset + start_idx, seed_len, gen_len, d_output
            );
            CHECK_CUDA(cudaDeviceSynchronize()); 
            CHECK_CUDA(cudaMemcpy(h_output, d_output, gen_len, cudaMemcpyDeviceToHost));
            
            printf("\033[32m");
            for(int i=0; i<seed_len; i++) {
                char c = ds.data[start_idx+i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[36m");
            for(int i=0; i<gen_len; i++) {
                char c = h_output[i];
                if(c>=32 && c<=126) printf("%c", c); else printf(".");
            }
            printf("\033[0m\n");

            double fwd_ms = get_time_diff_ms(t0, t1);
            double host_ms = get_time_diff_ms(t1, t2);
            double upd_ms = get_time_diff_ms(t2, t3);
            double step_ms = get_time_diff_ms(t0, t3);
            double steps_per_sec = (step_ms > 0) ? (1000.0 / step_ms) : 0.0;
            
            double total_t = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)/1e9;
            
            printf("Step %ld | Loss: %.4f | Up+: %d Up-: %d | Fwd: %.1fms Host: %.1fms Upd: %.1fms | %.2f Steps/s | Tok/s: %.2f\n", 
               step, current_loss_per_token, h_updates[0], h_updates[1], 
               fwd_ms, host_ms, upd_ms, steps_per_sec, total_tokens / total_t); 
        }
    }

    if (!keep_running) {
        printf("\nTraining interrupted by User. Exiting gracefully...\n");
    }

    printf("Cleaning up resources...\n");
    free(ds.data);
    free(h_model);
    
    cudaFree(d_model);
    cudaFree(d_dataset);
    cudaFree(d_accum_loss);
    cudaFree(d_fitnesses);
    cudaFree(d_pop_states);
    
    return 0;
}
