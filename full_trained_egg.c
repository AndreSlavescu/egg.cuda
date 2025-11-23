#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

// --- Configuration [cite: 275, 277, 288] ---
#define VOCAB_SIZE 256        // Byte-level tokenization
#define HIDDEN_DIM 128        // Model width
#define N_LAYERS 4            // Number of layers
#define SEQ_LEN 128            // Sequence length for BPTT (truncated)
#define POPULATION_SIZE 256    // Number of perturbations per step
#define BATCH_SIZE 128          // Parallel streams
#define FIXED_POINT 4         // 4 bits for fractional part
#define SIGMA_SHIFT 4         // Noise scale (bitwise shift)
#define UPDATE_THRESHOLD 100  // Votes needed to flip a weight [cite: 1023]
#define MAX_VAL 127
#define MIN_VAL -127

// --- Lookup Tables [cite: 998-1000] ---
// int32_t LOG2_TABLE[1024]; // Removed: insufficient size
int32_t EXP2_TABLE[256];

// --- Data Structure ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// --- Model Parameters Struct ---
// Storing all weights in a flat structure for easier iteration
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    // Layers: We will flatten them logic-wise into arrays for the update loop
    // 0: Wf, 1: Uf, 2: Wh, 3: Uh (GRU) | 4: W1, 5: W2 (MLP) | 6: Head
    // We simplify LN weights to be fixed or separately managed to reduce code size
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; // 0: bf, 1: bh
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; // 0: Expand, 1: Project
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM]; // 0: LN1, 1: LN2
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// --- Helper Functions ---

void init_tables() {
    for(int i=0; i<256; i++) 
        EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

// Saturated Addition [cite: 283, 936]
int8_t clipped_add(int32_t a, int32_t b) {
    int32_t res = a + b;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

// RNG helpers
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_val(uint32_t *rng) {
    return (int8_t)((xorshift32(rng) & 31) - 16);
}

// --- The Core "Rank-1 Perturbed Matrix Mul" [cite: 68, 1009] ---
// Computes x(W + s*AB^T)^T = xW^T + s(xB)A^T
// If noise_sign is 0, it performs standard inference.
void matmul_perturbed(
    const int8_t *in, const int8_t *w, int8_t *out, 
    int rows, int cols, 
    uint32_t layer_seed, int noise_sign
) {
    // 1. Generate Noise Vectors A and B on the fly
    // Optimization: Use Stack VLA instead of malloc
    int8_t A[rows];
    int8_t B[cols];
    
    uint32_t rng = layer_seed;
    for(int i=0; i<rows; i++) A[i] = gen_noise_val(&rng);
    for(int i=0; i<cols; i++) B[i] = gen_noise_val(&rng);

    // 2. Compute xB (Projection onto B)
    int32_t xB = 0;
    for(int j=0; j<cols; j++) xB += (int32_t)in[j] * (int32_t)B[j];

    // 3. Compute Result
    int shift = 8; // Division by 16*sqrt(n) approx
    for(int i=0; i<rows; i++) {
        int32_t acc = 0;
        for(int j=0; j<cols; j++) acc += (int32_t)in[j] * (int32_t)w[i * cols + j];
        
        if (noise_sign != 0) {
            int32_t noise = (xB * (int32_t)A[i]) * noise_sign;
            acc += (noise >> SIGMA_SHIFT);
        }
        
        int32_t res = acc >> shift;
        if(res > MAX_VAL) out[i] = MAX_VAL;
        else if(res < MIN_VAL) out[i] = MIN_VAL;
        else out[i] = (int8_t)res;
    }
}

// --- Layer Norm [cite: 965] ---
void egg_ln(const int8_t *x, const int8_t *w, int8_t *out) {
    int32_t sum = 0;
    for(int i=0; i<HIDDEN_DIM; i++) sum += abs(x[i]);
    if(sum == 0) sum = 1;
    int32_t mean = sum / HIDDEN_DIM;
    if(mean == 0) mean = 1;
    
    for(int i=0; i<HIDDEN_DIM; i++) {
        int32_t val = ((int32_t)x[i] * w[i]) / mean;
        if(val > MAX_VAL) out[i] = MAX_VAL;
        else if(val < MIN_VAL) out[i] = MIN_VAL;
        else out[i] = (int8_t)val;
    }
}

// --- Forward Pass (With Noise Injection) [cite: 915] ---
void forward_pass(
    EggModel *model, 
    const uint8_t *inputs, 
    int seq_len, 
    int8_t *logits_out, 
    uint32_t step_seed, 
    int noise_sign
) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM], state[N_LAYERS][HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM], buf2[HIDDEN_DIM], buf3[HIDDEN_DIM]; // scratch pads
    memset(state, 0, sizeof(state));

    for(int t=0; t<seq_len; t++) {
        // Embedding (Perturbed)
        // Simplification: We only perturb embedding via the linear projection logic if implemented as matmul
        // Here simple lookup:
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        // Layers
        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100); // Unique seed per layer

            // -- GRU --
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][0], x);

            // Wf, Uf -> f_gate
            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+1, noise_sign);
            matmul_perturbed(state[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+2, noise_sign);
            // ft = sigmoid(Wf + Uf + bf) -> Identity
            int8_t ft[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            // Gated Past
            int8_t gated_past[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * state[l][i]) >> 8);

            // Wh, Uh -> h_cand
            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+3, noise_sign);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+4, noise_sign);
            int8_t ht[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            // State Update
            for(int i=0; i<HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - state[l][i])) >> 8;
                state[l][i] = clipped_add(state[l][i], update);
                x[i] = state[l][i]; // Output is new state
            }
            
            // Residual Add
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            // -- MLP --
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);
            
            // Expansion (In: HIDDEN, Out: 4*HIDDEN)
            // We use a simplified MLP (Hidden -> Hidden) to save C-code complexity/memory here
            // or split the large matrix into chunks. 
            // For brevity: using Hidden->Hidden MLP.
            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+5, noise_sign);
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM, l_seed+6, noise_sign);

            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
        }

        // Final Head
        egg_ln(x, model->ln_out, x);
        // Compute logits only for the last token if seq_len > 1 (simplification for loss calc)
        // or for every token if doing full sequence training.
        // We output the last one for batch loss calculation.
        matmul_perturbed(x, model->head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, noise_sign);
    }
}

// --- Helper for Loss Calculation ---
static inline int get_msb(uint32_t n) {
    int pos = 0;
    if (n >= 1<<16) { n >>= 16; pos += 16; }
    if (n >= 1<<8)  { n >>= 8;  pos += 8; }
    if (n >= 1<<4)  { n >>= 4;  pos += 4; }
    if (n >= 1<<2)  { n >>= 2;  pos += 2; }
    if (n >= 1<<1)  {           pos += 1; }
    return pos;
}

// Returns 16 * log2(x / 16)
int32_t log2_fixed(int32_t x) {
    if (x <= 0) return 0; // Check for safety
    int k = get_msb(x);
    // We want 16 * log2(x) - 64
    // log2(x) approx k + (x - 2^k)/2^k
    // 16*log2(x) approx 16*k + 16*(x - 2^k)/2^k
    
    // fraction calculation safely:
    int32_t fraction;
    if (k >= 4) {
        fraction = (x - (1 << k)) >> (k - 4);
    } else {
        fraction = (x - (1 << k)) << (4 - k);
    }
    
    return (k << 4) + fraction - 64;
}

// --- Loss Calculation [cite: 994-997] ---
int32_t compute_loss(int8_t *logits, uint8_t target) {
    int32_t sum_exp = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        sum_exp += EXP2_TABLE[idx < 0 ? 0 : (idx > 255 ? 255 : idx)];
    }
    // Fix: Use full range log2 approximation instead of small table
    int32_t log_sum = log2_fixed(sum_exp);
    int32_t target_logit = (int32_t)logits[target] + 128;
    return log_sum - target_logit;
}

// --- Parameter Update (The "Vote") [cite: 1019-1023] ---
void update_matrix(
    int8_t *W, int rows, int cols, 
    uint32_t seed, 
    const int *fitnesses, // Array of fitness values per population member
    int pop_size
) {
    // Optimization: Use persistent buffer for votes and stack for noise vectors
    static int32_t votes[65536 * 4]; // Handle up to HIDDEN_DIM * (HIDDEN_DIM * 4)
    memset(votes, 0, rows * cols * sizeof(int32_t));

    int8_t A[rows];
    int8_t B[cols];

    // Reconstruct Noise and Accumulate Votes
    for(int p=0; p<pop_size; p+=2) {
        uint32_t p_seed = seed + (p/2); // Same seed logic as training loop
        uint32_t rng = p_seed;
        for(int i=0; i<rows; i++) A[i] = gen_noise_val(&rng);
        for(int i=0; i<cols; i++) B[i] = gen_noise_val(&rng);

        int f = fitnesses[p/2]; // Fitness for this antithetical pair
        
        // Vote = f * (AB^T)
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                votes[i*cols + j] += f * ((int32_t)A[i] * B[j]);
            }
        }
    }

    // Apply Update
    for(int i=0; i<rows*cols; i++) {
        if(votes[i] > UPDATE_THRESHOLD && W[i] < MAX_VAL) W[i]++;
        else if(votes[i] < -UPDATE_THRESHOLD && W[i] > MIN_VAL) W[i]--;
    }
}

// --- Sampling Helper ---
#define COLOR_GREEN "\033[32m"
#define COLOR_CYAN  "\033[36m"
#define COLOR_RESET "\033[0m"

int sample_logits(int8_t *logits) {
    int32_t probs[VOCAB_SIZE];
    int32_t sum = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        probs[i] = EXP2_TABLE[idx];
        sum += probs[i];
    }
    if(sum == 0) return 0;
    int32_t r = rand() % sum;
    int32_t acc = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        acc += probs[i];
        if(r < acc) return i;
    }
    return VOCAB_SIZE - 1;
}

void sample_model(EggModel *model, const uint8_t *seed_text, int seed_len, int gen_len) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM], state[N_LAYERS][HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM], buf2[HIDDEN_DIM];
    int8_t logits[VOCAB_SIZE];
    memset(state, 0, sizeof(state));

    printf(COLOR_GREEN);
    int input_token = 0;
    
    for(int t=0; t < seed_len + gen_len; t++) {
        if (t < seed_len) {
            input_token = seed_text[t];
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        } else {
            if (t == seed_len) printf(COLOR_CYAN);
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        }

        // -- Forward Layer Logic --
        memcpy(x, &model->embedding[input_token * HIDDEN_DIM], HIDDEN_DIM);

        for(int l=0; l<N_LAYERS; l++) {
            // GRU
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][0], x);

            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, 0, 0);
            matmul_perturbed(state[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, 0, 0);
            int8_t ft[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            int8_t gated_past[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * state[l][i]) >> 8);

            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, 0, 0);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, 0, 0);
            int8_t ht[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            for(int i=0; i<HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - state[l][i])) >> 8;
                state[l][i] = clipped_add(state[l][i], update);
                x[i] = state[l][i];
            }
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            // MLP
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);
            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, 0, 0);
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM, 0, 0);
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
        }

        // -- Predict Next Token --
        egg_ln(x, model->ln_out, x);
        matmul_perturbed(x, model->head, logits, VOCAB_SIZE, HIDDEN_DIM, 0, 0);
        
        if (t >= seed_len - 1) {
            input_token = sample_logits(logits);
        }
    }
    printf(COLOR_RESET "\n");
}

// --- Main Training Loop ---

Dataset load_data(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if(!f) { printf("Error: Create 'input.txt' first.\n"); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t*)malloc(len);
    fread(data, 1, len, f);
    fclose(f);
    return (Dataset){data, len};
}

int main() {
    srand(time(NULL));
    init_tables();
    Dataset ds = load_data("input.txt");
    printf("Loaded dataset: %ld bytes\n", ds.length);

    // Init Model (Random)
    EggModel *model = (EggModel*)calloc(1, sizeof(EggModel));
    uint32_t init_rng = 42;
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) model->embedding[i] = gen_noise_val(&init_rng);
    for(int i=0; i<HIDDEN_DIM*VOCAB_SIZE; i++) model->head[i] = gen_noise_val(&init_rng);
    // (Initialize other weights similarly... omitted for brevity)
    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16; // Init LN scale
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    }

    int *pair_fitnesses = (int*)malloc((POPULATION_SIZE/2) * sizeof(int));
    int8_t *logits = (int8_t*)malloc(VOCAB_SIZE);

    printf("Starting EGGROLL Training...\n");
    
    clock_t start_time = clock();
    long total_tokens = 0;
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step < max_steps; step++) {
        uint32_t step_seed = (uint32_t)time(NULL) + step;
        int start_idx = step * SEQ_LEN;
        
        // Visual Sampling
        if(step % 10 == 0) {
            sample_model(model, &ds.data[start_idx], 30, 30);
        }

        // Monitoring
        if(step % 10 == 0) {
            forward_pass(model, &ds.data[start_idx], SEQ_LEN, logits, step_seed, 0);
            int32_t loss_val = compute_loss(logits, ds.data[start_idx + SEQ_LEN]);
            double elapsed_sec = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            double tps = (elapsed_sec > 0) ? (double)total_tokens / elapsed_sec : 0.0;
            printf("Step %ld/%ld | Loss: %.4f | Tok/s: %.2f\n", step, max_steps, (double)loss_val / (1 << FIXED_POINT), tps);
        }

        // 1. Evaluate Population
        // We use antithetical pairs: (seed, +1) and (seed, -1)
        for(int p=0; p < POPULATION_SIZE; p+=2) {
            uint32_t p_seed = step_seed + (p/2); 
            
            // Pos Run
            forward_pass(model, &ds.data[start_idx], SEQ_LEN, logits, p_seed, 1);
            int32_t loss_pos = compute_loss(logits, ds.data[start_idx + SEQ_LEN]);
            
            // Neg Run
            forward_pass(model, &ds.data[start_idx], SEQ_LEN, logits, p_seed, -1);
            int32_t loss_neg = compute_loss(logits, ds.data[start_idx + SEQ_LEN]);

            // Fitness Shaping [cite: 1014-1017]
            if (loss_pos < loss_neg) pair_fitnesses[p/2] = 1;
            else if (loss_neg < loss_pos) pair_fitnesses[p/2] = -1;
            else pair_fitnesses[p/2] = 0;
        }

        // 2. Update Weights
        // Iterate over all layers and update
        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);
            update_matrix(model->gru_weights[l][0], HIDDEN_DIM, HIDDEN_DIM, l_seed+1, pair_fitnesses, POPULATION_SIZE); // Wf
            update_matrix(model->gru_weights[l][1], HIDDEN_DIM, HIDDEN_DIM, l_seed+2, pair_fitnesses, POPULATION_SIZE); // Uf
            update_matrix(model->gru_weights[l][2], HIDDEN_DIM, HIDDEN_DIM, l_seed+3, pair_fitnesses, POPULATION_SIZE); // Wh
            update_matrix(model->gru_weights[l][3], HIDDEN_DIM, HIDDEN_DIM, l_seed+4, pair_fitnesses, POPULATION_SIZE); // Uh
            
            update_matrix(model->mlp_weights[l][0], HIDDEN_DIM, HIDDEN_DIM, l_seed+5, pair_fitnesses, POPULATION_SIZE); // MLP1
            update_matrix(model->mlp_weights[l][1], HIDDEN_DIM, HIDDEN_DIM, l_seed+6, pair_fitnesses, POPULATION_SIZE); // MLP2
        }
        update_matrix(model->head, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, pair_fitnesses, POPULATION_SIZE);

        total_tokens += SEQ_LEN;
    }

    printf("Training Done.\n");
    free(ds.data); free(model); free(logits); free(pair_fitnesses);
    return 0;
}
