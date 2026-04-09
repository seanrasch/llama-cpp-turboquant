// CUDA port of TurboFlash: two-pass fused asymmetric attention for turbo3 V decode.
//
// Port of the Metal kernels kernel_turbo_flash_p1 / kernel_turbo_flash_p2
// introduced in TheTom/llama-cpp-turboquant@0d6b38aad. Architecture mirrors
// the Metal implementation so the two backends stay in lockstep:
//
//   Pass 1 (turbo_flash_p1): one warp (32 threads) per (query-head, KV-block).
//     - Q loaded to registers, each lane owns DK/32 interleaved dims.
//     - K dequant inline (turbo3 or q8_0 via template specialization).
//     - V dequant inline (turbo3, 3-bit centroid + sign bit, norm scale).
//     - Warp-level Q·K via __shfl_xor_sync butterfly reduction.
//     - Online softmax (m, l, o) entirely in registers, no shared memory.
//     - Each block writes {partial_output[DV], block_max, block_sum}.
//
//   Pass 2 (turbo_flash_p2): one block of DV threads per query head.
//     - Thread 0 scans partial_ms to compute global max and corrected sum.
//     - All DV threads merge partial outputs with softmax correction.
//     - Inverse WHT: s2 signs → butterfly (intra-warp via shuffle, cross-warp
//       via shared memory) → s1 signs × 1/sqrt(DV).
//     - Writes final dst[bh_idx * DV + d].
//
// Eligibility (matches Metal gate):
//   - Single-token decode: ne01 == 1
//   - V type: GGML_TYPE_TURBO3_0
//   - K type: GGML_TYPE_Q8_0 or GGML_TYPE_TURBO3_0
//   - Head dim: 64, 96, or 128 (multiple of 32)
//   - Environment: TURBO_FLASH={0 (off), 1 (force), unset (default on)}

#pragma once

#include "common.cuh"
#include "turbo-quant.cuh"

// Host-side API exposed by fattn-turbo-flash.cu.
// The CUDA FA dispatcher in fattn.cu calls these two functions.
bool ggml_cuda_flash_attn_ext_use_turbo_flash(const ggml_tensor * op);
void ggml_cuda_flash_attn_ext_turbo_flash(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#define CUDA_TURBO_FLASH_BLOCK_SIZE 64
#define CUDA_TURBO_FLASH_WARP_SIZE  32

// TURBO_WHT_SIGNS1 / TURBO_WHT_SIGNS2 / TURBO_CENTROIDS_3BIT come in via the
// `#include "turbo-quant.cuh"` above. They're declared `static __constant__`,
// so each .cu file that includes this header gets its own device-side copy —
// this is the same pattern the existing CUDA turbo path uses.

// ─── Pass 1 ──────────────────────────────────────────────────────────────────
//
// Grid:        (n_bh, n_blocks, 1)    — n_bh = ne01 * ne02 * ne03
// Block:       (32, 1, 1)             — exactly one warp
// Shared mem:  0 bytes
//
// Each warp processes one (query-head, KV-block) pair. Within the warp, each
// lane handles DK/32 interleaved dimensions on the K side and DV/32 on the V
// side (e.g., for DK=DV=128: lane 0 owns dims {0, 32, 64, 96}).
//
// The token loop runs over up to CUDA_TURBO_FLASH_BLOCK_SIZE (64) tokens from
// the block's KV slice, updating the online softmax state on every step.
//
template <int DK, int DV, bool K_IS_TURBO3, bool HAS_MASK>
__launch_bounds__(CUDA_TURBO_FLASH_WARP_SIZE, 2)
__global__ void turbo_flash_p1_kernel(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        float      * __restrict__ partial_out,
        float      * __restrict__ partial_ms,
        const int      ne01,
        const int      ne02,
        const int      ne03,
        const uint64_t nb01,
        const uint64_t nb02,
        const uint64_t nb03,
        const int      ne11,
        const int      ne_12_2,
        const int      ne_12_3,
        const uint64_t nb11,
        const uint64_t nb12,
        const uint64_t nb13,
        const uint64_t nb21,
        const uint64_t nb22,
        const uint64_t nb23,
        const int      ne32,
        const int      ne33,
        const uint64_t nb31,
        const uint64_t nb32,
        const uint64_t nb33,
        const float    scale,
        const int      n_blocks) {

    static_assert(DK % 32 == 0, "DK must be a multiple of 32");
    static_assert(DV % 32 == 0, "DV must be a multiple of 32");

    constexpr int DK_PER_LANE = DK / 32;
    constexpr int DV_PER_LANE = DV / 32;

    const int lane     = threadIdx.x;      // 0..31
    const int bh_idx   = blockIdx.x;
    const int block_id = blockIdx.y;

    const int T_kv    = ne11;
    const int t_start = block_id * CUDA_TURBO_FLASH_BLOCK_SIZE;
    const int t_end   = min(t_start + CUDA_TURBO_FLASH_BLOCK_SIZE, T_kv);

    // Decompose bh_idx → (iq1, iq2, iq3) for pointer arithmetic
    const int iq1 = bh_idx % ne01;
    const int iq2 = (bh_idx / ne01) % ne02;
    const int iq3 = bh_idx / (ne01 * ne02);

    // GQA: map query head → KV head
    const int ikv2 = iq2 / (ne02 / ne_12_2);
    const int ikv3 = iq3 / (ne03 / ne_12_3);

    // ─── Load Q into registers ───────────────────────────────────────────
    const float * q_ptr = (const float *)(q + iq1*nb01 + iq2*nb02 + iq3*nb03);
    float q_vals[DK_PER_LANE];
#pragma unroll
    for (int i = 0; i < DK_PER_LANE; i++) {
        const int d = lane + i * 32;
        q_vals[i] = (d < DK) ? q_ptr[d] : 0.0f;
    }

    // ─── Online softmax state (registers only) ───────────────────────────
    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state[DV_PER_LANE];
#pragma unroll
    for (int i = 0; i < DV_PER_LANE; i++) {
        o_state[i] = 0.0f;
    }

    // ─── K / V base pointers for this KV head ────────────────────────────
    const char * k_base = k + ikv2*nb12 + ikv3*nb13;
    const char * v_base = v + ikv2*nb22 + ikv3*nb23;

    const half * mask_ptr = nullptr;
    if (HAS_MASK) {
        mask_ptr = (const half *)(mask + iq1*nb31 + (iq2 % ne32)*nb32 + (iq3 % ne33)*nb33);
    }

    // ─── Token loop ──────────────────────────────────────────────────────
    for (int t = t_start; t < t_end; t++) {

        // Mask early-out
        float mask_val = 0.0f;
        if (HAS_MASK) {
            mask_val = __half2float(mask_ptr[t]);
            if (mask_val <= -65504.0f) {
                continue;
            }
        }

        // Dequant K and compute per-lane partial dot product
        float dot_partial = 0.0f;

        if (K_IS_TURBO3) {
            const block_turbo3_0 * k_row = (const block_turbo3_0 *)(k_base + t * nb11);
            const float k_norm = __half2float(k_row[0].norm);

#pragma unroll
            for (int i = 0; i < DK_PER_LANE; i++) {
                const int d = lane + i * 32;
                if (d >= DK) break;

                const int     qs_byte  = d / 4;
                const int     qs_shift = (d % 4) * 2;
                const uint8_t qi       = (k_row[0].qs[qs_byte] >> qs_shift) & 0x3;

                const int     sign_byte = d / 8;
                const int     sign_bit  = d % 8;
                const uint8_t sb        = (k_row[0].signs[sign_byte] >> sign_bit) & 1;

                const uint8_t idx = qi | (sb << 2);
                dot_partial += q_vals[i] * TURBO_CENTROIDS_3BIT[idx] * k_norm;
            }
        } else {
            // K is q8_0: 32 elements per block, DK/32 blocks per row.
            const block_q8_0 * k_row = (const block_q8_0 *)(k_base + t * nb11);

#pragma unroll
            for (int i = 0; i < DK_PER_LANE; i++) {
                const int d = lane + i * 32;
                if (d >= DK) break;

                const int qb = d / 32;
                const int qj = d % 32;
                dot_partial += q_vals[i] * (float)k_row[qb].qs[qj] * __half2float(k_row[qb].d);
            }
        }

        // Warp-level reduction (butterfly)
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot_partial += __shfl_xor_sync(0xffffffff, dot_partial, offset);
        }
        const float score = dot_partial * scale + mask_val;

        // Dequant V (turbo3 only in this kernel)
        const block_turbo3_0 * v_row = (const block_turbo3_0 *)(v_base + t * nb21);
        const float v_norm = __half2float(v_row[0].norm);

        float v_decoded[DV_PER_LANE];
#pragma unroll
        for (int i = 0; i < DV_PER_LANE; i++) {
            const int d = lane + i * 32;
            if (d >= DV) { v_decoded[i] = 0.0f; continue; }

            const int     qs_byte  = d / 4;
            const int     qs_shift = (d % 4) * 2;
            const uint8_t qi       = (v_row[0].qs[qs_byte] >> qs_shift) & 0x3;

            const int     sign_byte = d / 8;
            const int     sign_bit  = d % 8;
            const uint8_t sb        = (v_row[0].signs[sign_byte] >> sign_bit) & 1;

            const uint8_t idx = qi | (sb << 2);
            v_decoded[i] = TURBO_CENTROIDS_3BIT[idx] * v_norm;
        }

        // Online softmax update (Dao et al. Flash-Attention-2 style)
        const float new_m    = fmaxf(m_state, score);
        const float exp_diff = __expf(m_state - new_m);
        const float exp_score = __expf(score    - new_m);

#pragma unroll
        for (int i = 0; i < DV_PER_LANE; i++) {
            o_state[i] = o_state[i] * exp_diff + exp_score * v_decoded[i];
        }
        l_state = l_state * exp_diff + exp_score;
        m_state = new_m;
    }

    // ─── Write partial results ───────────────────────────────────────────
    // Scatter write — each lane owns its own dims, no reduction needed.
#pragma unroll
    for (int i = 0; i < DV_PER_LANE; i++) {
        const int d = lane + i * 32;
        if (d < DV) {
            partial_out[(uint64_t)bh_idx * n_blocks * DV
                        + (uint64_t)block_id * DV
                        + d] = o_state[i];
        }
    }

    // Lane 0 writes the block's max and exp-sum.
    if (lane == 0) {
        partial_ms[(uint64_t)bh_idx * n_blocks * 2 + (uint64_t)block_id * 2 + 0] = m_state;
        partial_ms[(uint64_t)bh_idx * n_blocks * 2 + (uint64_t)block_id * 2 + 1] = l_state;
    }
}

// ─── Pass 2 ──────────────────────────────────────────────────────────────────
//
// Grid:        (n_bh, 1, 1)
// Block:       (DV, 1, 1)
// Shared mem:  (DV + 2) * sizeof(float)
//
// Merges the per-block partial outputs from pass 1 using online-softmax
// correction, then applies the inverse WHT rotation (s2 → butterfly → s1).
// The butterfly uses warp shuffles for the first 5 stages (when DV ≥ 32) and
// shared memory for the last 2 stages (cross-warp, DV > 32).
//
template <int DV>
__launch_bounds__(DV, 4)
__global__ void turbo_flash_p2_kernel(
        const float * __restrict__ partial_out,
        const float * __restrict__ partial_ms,
        float       * __restrict__ dst,
        const int n_blocks) {

    // DV==96 would need a hybrid WHT group (rotation group is always a power
    // of two — 64 or 128). For now restrict to clean power-of-two DV; add a
    // tail path if a 96-dim model becomes interesting.
    static_assert(DV == 64 || DV == 128, "DV must be 64 or 128");

    extern __shared__ float s_mem[];
    float * shared_out = s_mem;                 // [DV]
    float * shared_gmax = s_mem + DV;           // [1]
    float * shared_gsum = s_mem + DV + 1;       // [1]

    const int tid    = threadIdx.x;
    const int bh_idx = blockIdx.x;

    // ─── Step 1: global max + corrected sum (thread 0, serial) ───────────
    if (tid == 0) {
        float gmax = -INFINITY;
        for (int b = 0; b < n_blocks; b++) {
            const float bmax = partial_ms[(uint64_t)bh_idx * n_blocks * 2 + (uint64_t)b * 2 + 0];
            gmax = fmaxf(gmax, bmax);
        }
        shared_gmax[0] = gmax;

        float gsum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const float bmax = partial_ms[(uint64_t)bh_idx * n_blocks * 2 + (uint64_t)b * 2 + 0];
            const float bsum = partial_ms[(uint64_t)bh_idx * n_blocks * 2 + (uint64_t)b * 2 + 1];
            gsum += __expf(bmax - gmax) * bsum;
        }
        shared_gsum[0] = gsum;
    }
    __syncthreads();

    const float global_max     = shared_gmax[0];
    const float global_sum     = shared_gsum[0];
    const float inv_global_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    // ─── Step 2: merge partial outputs with softmax correction ───────────
    if (tid < DV) {
        float accum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const float bmax       = partial_ms[(uint64_t)bh_idx * n_blocks * 2 + (uint64_t)b * 2 + 0];
            const float correction = __expf(bmax - global_max);
            const float block_val  = partial_out[(uint64_t)bh_idx * n_blocks * DV
                                                 + (uint64_t)b * DV + tid];
            accum += correction * block_val;
        }
        shared_out[tid] = accum * inv_global_sum;
    }
    __syncthreads();

    // ─── Step 3: Inverse WHT (s2 → butterfly → s1 × 1/√DV) ───────────────
    // Only the first min(DV, 128) elements participate in the rotation group.
    if (tid < DV && tid < 128) {
        // 3a: s2 signs
        float val = shared_out[tid] * TURBO_WHT_SIGNS2[tid];

        // 3b: butterfly — first up-to-5 stages via warp shuffle (intra-warp).
        //     log2(DV): 128→7, 96→? (non-power-of-2, skip), 64→6, 32→5.
        //     DV=96 is handled by a separate scalar fallback (3 stages cover 32).
        constexpr int log2_dv = (DV >= 128) ? 7 : (DV >= 64) ? 6 : 5;
        constexpr int simd_stages = (log2_dv < 5) ? log2_dv : 5;

        const int lane_in_warp = tid % 32;
#pragma unroll
        for (int s = 0; s < simd_stages; s++) {
            const int  step = 1 << s;
            const float other = __shfl_xor_sync(0xffffffff, val, step);
            val = (lane_in_warp & step) ? (other - val) : (other + val);
        }

        // 3c: remaining cross-warp stages via shared memory (DV > 32)
        if (DV > 32) {
            shared_out[tid] = val;
            __syncthreads();

            for (int half_block = 32; half_block < DV; half_block <<= 1) {
                const int bfly_size = half_block << 1;
                const int bfly_idx  = tid / bfly_size;
                const int local_idx = tid % bfly_size;
                const int base_idx  = bfly_idx * bfly_size;

                const float a = shared_out[base_idx + (local_idx % half_block)];
                const float b = shared_out[base_idx + (local_idx % half_block) + half_block];
                __syncthreads();

                shared_out[tid] = (local_idx < half_block) ? (a + b) : (a - b);
                __syncthreads();
            }
            val = shared_out[tid];
        }

        // 3d: s1 signs + 1/√DV normalization
        const float inv_sqrt_dim = rsqrtf((float)DV);
        dst[(uint64_t)bh_idx * DV + tid] = val * inv_sqrt_dim * TURBO_WHT_SIGNS1[tid];
    }
}
