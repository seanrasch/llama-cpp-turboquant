// CUDA TurboFlash: two-pass fused asymmetric attention launch glue.
// See fattn-turbo-flash.cuh for kernel details and architecture notes.

#include "fattn-turbo-flash.cuh"
#include "fattn-common.cuh"

#include <cstdlib>
#include <cstring>

// Eligibility mirror of Metal's ggml_metal_op_flash_attn_ext_use_turbo_flash.
// Returns true when the caller should route this FA op through TurboFlash.
bool ggml_cuda_flash_attn_ext_use_turbo_flash(const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne01   = op->src[0]->ne[1];  // queries per head (batch size)
    const int64_t ne00   = op->src[0]->ne[0];  // head dim

    const ggml_type type_k = op->src[1]->type;
    const ggml_type type_v = op->src[2]->type;

    // Single-token decode only.
    if (ne01 != 1) return false;

    // V must be turbo3.
    if (type_v != GGML_TYPE_TURBO3_0) return false;

    // K is either q8_0 (asymmetric) or turbo3 (symmetric).
    if (type_k != GGML_TYPE_Q8_0 && type_k != GGML_TYPE_TURBO3_0) return false;

    // Matching K/V head size, currently only DK=DV=128 is ported.
    if (op->src[2]->ne[0] != ne00) return false;
    if (ne00 != 128) return false;

    // Bail on features TurboFlash does not implement yet. Unlike the Metal
    // gate, we refuse these explicitly rather than silently produce wrong
    // outputs — the generic VEC kernel will handle them correctly.
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(float));
    if (max_bias != 0.0f)      return false;  // ALiBi unsupported
    if (logit_softcap != 0.0f) return false;  // Gemma-style softcap unsupported
    if (op->src[4] != nullptr) return false;  // attention sinks unsupported

    // Environment kill-switch (mirrors Metal: "0"=off, "1"=force, default=on).
    const char * env = getenv("TURBO_FLASH");
    if (env && env[0] == '0') return false;

    return true;
}

// Dispatcher-side entry. Allocates pass-1 scratch from the CUDA pool and
// launches the two kernels sequentially on the op's main stream.
void ggml_cuda_flash_attn_ext_turbo_flash(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q->ne[0] == 128 && V->ne[0] == 128);  // current scope
    GGML_ASSERT(Q->ne[1] == 1);                        // single-token decode
    GGML_ASSERT(V->type == GGML_TYPE_TURBO3_0);
    GGML_ASSERT(K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    constexpr int DK = 128;
    constexpr int DV = 128;

    const int ne01 = (int) Q->ne[1];
    const int ne02 = (int) Q->ne[2];
    const int ne03 = (int) Q->ne[3];

    const int ne11 = (int) K->ne[1];     // T_kv
    const int ne_12_2 = (int) K->ne[2];  // number of KV heads dim 2
    const int ne_12_3 = (int) K->ne[3];  // number of KV heads dim 3

    const int n_bh     = ne01 * ne02 * ne03;
    const int n_blocks = (ne11 + CUDA_TURBO_FLASH_BLOCK_SIZE - 1) / CUDA_TURBO_FLASH_BLOCK_SIZE;

    // Attention scale: the graph stores it at op_params[0].
    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream   = ctx.stream();

    ggml_cuda_pool_alloc<float> partial_out(pool, (size_t) n_bh * n_blocks * DV);
    ggml_cuda_pool_alloc<float> partial_ms (pool, (size_t) n_bh * n_blocks * 2);

    const char * q_data = (const char *) Q->data;
    const char * k_data = (const char *) K->data;
    const char * v_data = (const char *) V->data;
    const char * m_data = mask ? (const char *) mask->data : nullptr;

    const bool k_is_turbo3 = (K->type == GGML_TYPE_TURBO3_0);
    const bool has_mask    = (mask != nullptr);

    // Mask strides (only meaningful when mask != nullptr)
    const int      ne32 = mask ? (int) mask->ne[2] : 0;
    const int      ne33 = mask ? (int) mask->ne[3] : 0;
    const uint64_t nb31 = mask ? mask->nb[1] : 0;
    const uint64_t nb32 = mask ? mask->nb[2] : 0;
    const uint64_t nb33 = mask ? mask->nb[3] : 0;

    // ─── Pass 1 launch ───────────────────────────────────────────────────
    const dim3 p1_grid(n_bh, n_blocks, 1);
    const dim3 p1_block(CUDA_TURBO_FLASH_WARP_SIZE, 1, 1);

#define LAUNCH_P1(K_IS_TURBO3, HAS_MASK)                                    \
    turbo_flash_p1_kernel<DK, DV, K_IS_TURBO3, HAS_MASK>                    \
        <<<p1_grid, p1_block, 0, stream>>>(                                 \
            q_data, k_data, v_data, m_data,                                 \
            partial_out.ptr, partial_ms.ptr,                                \
            ne01, ne02, ne03,                                               \
            Q->nb[1], Q->nb[2], Q->nb[3],                                   \
            ne11, ne_12_2, ne_12_3,                                         \
            K->nb[1], K->nb[2], K->nb[3],                                   \
            V->nb[1], V->nb[2], V->nb[3],                                   \
            ne32, ne33, nb31, nb32, nb33,                                   \
            scale, n_blocks)

    if (k_is_turbo3 && has_mask)       { LAUNCH_P1(true,  true ); }
    else if (k_is_turbo3 && !has_mask) { LAUNCH_P1(true,  false); }
    else if (!k_is_turbo3 && has_mask) { LAUNCH_P1(false, true ); }
    else                               { LAUNCH_P1(false, false); }
#undef LAUNCH_P1
    CUDA_CHECK(cudaGetLastError());

    // ─── Pass 2 launch ───────────────────────────────────────────────────
    const dim3 p2_grid(n_bh, 1, 1);
    const dim3 p2_block(DV, 1, 1);
    const size_t p2_smem = (DV + 2) * sizeof(float);

    turbo_flash_p2_kernel<DV><<<p2_grid, p2_block, p2_smem, stream>>>(
        partial_out.ptr,
        partial_ms.ptr,
        (float *) dst->data,
        n_blocks);
    CUDA_CHECK(cudaGetLastError());
}
