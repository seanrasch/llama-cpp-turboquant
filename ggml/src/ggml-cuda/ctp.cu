/*
 * Compressed Tensor Parallelism (CTP)
 *
 * Quantizes activation tensors to q8_0 before cross-device PCIe transfers.
 * q8_0: 32 int8 values + fp32 scale per block = 34 bytes per 32 elements.
 * fp16 baseline: 64 bytes per 32 elements. Compression ratio: 1.88x.
 *
 * The quantization error is non-cumulative — each activation is consumed
 * once and discarded, unlike KV cache which accumulates error over the
 * sequence length.
 */

#include "ctp.cuh"
#include <cstdlib>
#include <cstdio>

// ---- Configuration ----

// Minimum tensor size to bother compressing (below this, raw copy is faster)
static const size_t CTP_MIN_BYTES = 32 * 1024;  // 32 KB

// ---- CTP enable check ----

bool ggml_cuda_ctp_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_CUDA_CTP");
        enabled = (env && atoi(env) != 0) ? 1 : 0;
        if (enabled) {
            fprintf(stderr, "CTP: compressed tensor parallelism enabled (q8_0 cross-device transfers)\n");
        }
    }
    return enabled != 0;
}

// ---- Quantize kernel: fp16 → q8_0 ----
// Each block of 32 threads handles one q8_0 block (32 elements).

static __global__ void k_ctp_quantize_f16_q8_0(
        const half * __restrict__ src,
        int8_t * __restrict__ dst_qs,
        float  * __restrict__ dst_d,
        int64_t n_blocks) {

    const int64_t ib = (int64_t)blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    if (ib >= n_blocks) return;

    const int lane = threadIdx.x % 32;
    const half * sp = src + ib * 32;

    // Find max absolute value across the block via warp reduction
    float val = __half2float(sp[lane]);
    float aval = fabsf(val);

    for (int offset = 16; offset > 0; offset >>= 1) {
        aval = fmaxf(aval, __shfl_xor_sync(0xffffffff, aval, offset));
    }

    // Compute scale and quantize
    const float d = aval / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / aval : 0.0f;

    dst_qs[ib * 32 + lane] = (int8_t)roundf(val * id);

    if (lane == 0) {
        dst_d[ib] = d;
    }
}

// ---- Quantize kernel: fp32 → q8_0 ----

static __global__ void k_ctp_quantize_f32_q8_0(
        const float * __restrict__ src,
        int8_t * __restrict__ dst_qs,
        float  * __restrict__ dst_d,
        int64_t n_blocks) {

    const int64_t ib = (int64_t)blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    if (ib >= n_blocks) return;

    const int lane = threadIdx.x % 32;
    const float * sp = src + ib * 32;

    float val = sp[lane];
    float aval = fabsf(val);

    for (int offset = 16; offset > 0; offset >>= 1) {
        aval = fmaxf(aval, __shfl_xor_sync(0xffffffff, aval, offset));
    }

    const float d = aval / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / aval : 0.0f;

    dst_qs[ib * 32 + lane] = (int8_t)roundf(val * id);

    if (lane == 0) {
        dst_d[ib] = d;
    }
}

// ---- Dequantize kernel: q8_0 → fp16 ----

static __global__ void k_ctp_dequantize_q8_0_f16(
        const int8_t * __restrict__ src_qs,
        const float  * __restrict__ src_d,
        half * __restrict__ dst,
        int64_t n_blocks) {

    const int64_t ib = (int64_t)blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    if (ib >= n_blocks) return;

    const int lane = threadIdx.x % 32;
    const float d = src_d[ib];
    dst[ib * 32 + lane] = __float2half(d * (float)src_qs[ib * 32 + lane]);
}

// ---- Dequantize kernel: q8_0 → fp32 ----

static __global__ void k_ctp_dequantize_q8_0_f32(
        const int8_t * __restrict__ src_qs,
        const float  * __restrict__ src_d,
        float * __restrict__ dst,
        int64_t n_blocks) {

    const int64_t ib = (int64_t)blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    if (ib >= n_blocks) return;

    const int lane = threadIdx.x % 32;
    const float d = src_d[ib];
    dst[ib * 32 + lane] = d * (float)src_qs[ib * 32 + lane];
}

// ---- Main CTP copy function ----

bool ggml_cuda_ctp_copy_tensor(
        int src_device, int dst_device,
        const void * src_data, void * dst_data,
        ggml_type src_type,
        int64_t nelements,
        cudaStream_t src_stream, cudaStream_t dst_stream) {

    // Only compress fp16 and fp32 tensors
    size_t element_size;
    if (src_type == GGML_TYPE_F16) {
        element_size = sizeof(half);
    } else if (src_type == GGML_TYPE_F32) {
        element_size = sizeof(float);
    } else {
        return false;
    }

    const size_t src_bytes = nelements * element_size;

    // Skip small tensors — raw copy is faster than quantize + transfer + dequantize
    if (src_bytes < CTP_MIN_BYTES) {
        return false;
    }

    // Pad element count to multiple of 32 for q8_0 blocks
    const int64_t n_blocks = (nelements + 31) / 32;
    const int64_t nelements_padded = n_blocks * 32;

    // Compressed buffer sizes
    const size_t qs_bytes = nelements_padded * sizeof(int8_t);
    const size_t d_bytes  = n_blocks * sizeof(float);
    const size_t compressed_bytes = qs_bytes + d_bytes;

    // Only proceed if we're actually saving bandwidth
    if (compressed_bytes >= src_bytes) {
        return false;
    }

    // Allocate temporary compressed buffers on source device.
    // If allocation fails (tight VRAM), fall back to raw copy silently.
    ggml_cuda_set_device(src_device);
    int8_t * src_compressed_qs = nullptr;
    float  * src_compressed_d  = nullptr;
    if (cudaMallocAsync(&src_compressed_qs, qs_bytes, src_stream) != cudaSuccess) {
        cudaGetLastError(); // clear error
        return false;
    }
    if (cudaMallocAsync(&src_compressed_d, d_bytes, src_stream) != cudaSuccess) {
        cudaGetLastError();
        cudaFreeAsync(src_compressed_qs, src_stream);
        return false;
    }

    // Quantize on source device
    const int threads_per_block = 256;
    const int blocks = (int)((n_blocks * 32 + threads_per_block - 1) / threads_per_block);

    if (src_type == GGML_TYPE_F16) {
        k_ctp_quantize_f16_q8_0<<<blocks, threads_per_block, 0, src_stream>>>(
            (const half *)src_data, src_compressed_qs, src_compressed_d, n_blocks);
    } else {
        k_ctp_quantize_f32_q8_0<<<blocks, threads_per_block, 0, src_stream>>>(
            (const float *)src_data, src_compressed_qs, src_compressed_d, n_blocks);
    }

    // Allocate receiving buffers on destination device
    ggml_cuda_set_device(dst_device);
    int8_t * dst_compressed_qs = nullptr;
    float  * dst_compressed_d  = nullptr;
    if (cudaMallocAsync(&dst_compressed_qs, qs_bytes, dst_stream) != cudaSuccess ||
        cudaMallocAsync(&dst_compressed_d,  d_bytes,  dst_stream) != cudaSuccess) {
        cudaGetLastError();
        // Can't allocate on dest — wait for src quantize to finish then free src buffers
        ggml_cuda_set_device(src_device);
        cudaStreamSynchronize(src_stream);
        cudaFreeAsync(src_compressed_qs, src_stream);
        cudaFreeAsync(src_compressed_d,  src_stream);
        if (dst_compressed_qs) { ggml_cuda_set_device(dst_device); cudaFreeAsync(dst_compressed_qs, dst_stream); }
        return false;
    }

    // Wait for quantization to complete before transferring
    cudaEvent_t quant_done;
    ggml_cuda_set_device(src_device);
    CUDA_CHECK(cudaEventCreateWithFlags(&quant_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(quant_done, src_stream));

    ggml_cuda_set_device(dst_device);
    CUDA_CHECK(cudaStreamWaitEvent(dst_stream, quant_done, 0));

    // Transfer compressed data across PCIe
    CUDA_CHECK(cudaMemcpyPeerAsync(dst_compressed_qs, dst_device, src_compressed_qs, src_device, qs_bytes, dst_stream));
    CUDA_CHECK(cudaMemcpyPeerAsync(dst_compressed_d,  dst_device, src_compressed_d,  src_device, d_bytes,  dst_stream));

    // Dequantize on destination device
    if (src_type == GGML_TYPE_F16) {
        k_ctp_dequantize_q8_0_f16<<<blocks, threads_per_block, 0, dst_stream>>>(
            dst_compressed_qs, dst_compressed_d, (half *)dst_data, n_blocks);
    } else {
        k_ctp_dequantize_q8_0_f32<<<blocks, threads_per_block, 0, dst_stream>>>(
            dst_compressed_qs, dst_compressed_d, (float *)dst_data, n_blocks);
    }

    // Free temporary buffers after dequant completes
    CUDA_CHECK(cudaFreeAsync(dst_compressed_qs, dst_stream));
    CUDA_CHECK(cudaFreeAsync(dst_compressed_d,  dst_stream));

    // Free source buffers after transfer completes
    cudaEvent_t transfer_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&transfer_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(transfer_done, dst_stream));

    ggml_cuda_set_device(src_device);
    CUDA_CHECK(cudaStreamWaitEvent(src_stream, transfer_done, 0));
    CUDA_CHECK(cudaFreeAsync(src_compressed_qs, src_stream));
    CUDA_CHECK(cudaFreeAsync(src_compressed_d,  src_stream));

    // Cleanup events
    CUDA_CHECK(cudaEventDestroy(quant_done));
    CUDA_CHECK(cudaEventDestroy(transfer_done));

    return true;
}
