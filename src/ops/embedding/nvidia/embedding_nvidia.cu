#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"

#include <algorithm>
#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void embeddingKernel(T *out, const int64_t *index, const T *weight, size_t num_indices, size_t hidden_size) {
    const size_t total = num_indices * hidden_size;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t linear_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; linear_idx < total; linear_idx += stride) {
        const size_t row = linear_idx / hidden_size;
        const size_t col = linear_idx % hidden_size;
        const int64_t vocab_row = index[row];
        out[linear_idx] = weight[static_cast<size_t>(vocab_row) * hidden_size + col];
    }
}

} // namespace

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t num_indices,
               size_t hidden_size, llaisysStream_t stream) {
    const size_t total = num_indices * hidden_size;
    if (total == 0) {
        return;
    }

    constexpr int threads = 256;
    const int max_blocks = 4096;
    const int blocks = static_cast<int>(std::min<size_t>((total + threads - 1) / threads, static_cast<size_t>(max_blocks)));
    const auto *index_ptr = reinterpret_cast<const int64_t *>(index);
    const cudaStream_t cuda_stream = llaisys::device::nvidia::toCudaStream(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embeddingKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<float *>(out),
            index_ptr,
            reinterpret_cast<const float *>(weight),
            num_indices,
            hidden_size);
        break;
    case LLAISYS_DTYPE_F16:
        embeddingKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::fp16_t *>(out),
            index_ptr,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            num_indices,
            hidden_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embeddingKernel<<<blocks, threads, 0, cuda_stream>>>(
            reinterpret_cast<llaisys::bf16_t *>(out),
            index_ptr,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            num_indices,
            hidden_size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
