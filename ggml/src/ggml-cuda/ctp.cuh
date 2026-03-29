/*
 * Compressed Tensor Parallelism (CTP)
 *
 * Compresses activation tensors before cross-device PCIe transfers,
 * reducing bandwidth by ~2x (q8_0) with negligible quality loss.
 * Activations are quantized on the source GPU, transferred compressed,
 * and dequantized on the destination GPU.
 *
 * Enable: GGML_CUDA_CTP=1 environment variable
 * Minimum tensor size to compress: GGML_CUDA_CTP_MIN_BYTES (default 32KB)
 */

#pragma once

#include "common.cuh"

// Check if CTP is enabled (cached after first call)
bool ggml_cuda_ctp_enabled(void);

// Perform a compressed cross-device tensor copy.
// Returns true if compression was applied, false if the copy should
// fall back to the standard cudaMemcpyPeerAsync path.
//
// The function:
//   1. Quantizes src to q8_0 on the source device
//   2. Transfers the compressed buffer to the dest device via cudaMemcpyPeerAsync
//   3. Dequantizes on the dest device back to the original type
//
// Only activates for fp16/fp32 tensors above the minimum size threshold.
bool ggml_cuda_ctp_copy_tensor(
    int src_device, int dst_device,
    const void * src_data, void * dst_data,
    ggml_type src_type,
    int64_t nelements,
    cudaStream_t src_stream, cudaStream_t dst_stream);
