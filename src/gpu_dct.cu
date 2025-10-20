#include "../include/gpu_dct.cuh"
#include "gpu_dct_impl.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <mutex>
#include <vector>

namespace gpu_dct {

namespace {

template<typename ComputeType>
void fill_dct_transform(ComputeType* out, int n)
{
    const ComputeType pi = static_cast<ComputeType>(3.14159265358979323846);
    for (int i = 0; i < n; ++i) {
        const ComputeType scale = (i == 0)
                                      ? std::sqrt(static_cast<ComputeType>(1) / n)
                                      : std::sqrt(static_cast<ComputeType>(2) / n);
        for (int j = 0; j < n; ++j) {
            out[static_cast<size_t>(i) * n + j] =
                scale * std::cos(pi * i * (2 * j + 1) / (2 * n));
        }
    }
}

template<typename ComputeType, int N>
std::array<ComputeType, static_cast<size_t>(N) * N> make_dct_transform_array()
{
    std::array<ComputeType, static_cast<size_t>(N) * N> result{};
    fill_dct_transform(result.data(), N);
    return result;
}

template<typename ComputeType>
std::vector<ComputeType> make_dct_transform_vector(int n)
{
    std::vector<ComputeType> result(static_cast<size_t>(n) * static_cast<size_t>(n));
    fill_dct_transform(result.data(), n);
    return result;
}

} // namespace

// ============================================================================
// Template Class Implementation
// ============================================================================

void throw_on_cuda_error(cudaError_t error, std::string_view context)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(error));
    }
}

template<typename T>
    requires DctScalar<T>
GpuDct<T>::GpuDct(int n, cudaStream_t stream)
    : m_size(n)
    , m_stream(stream)
    , m_owns_stream(stream == nullptr)
    , m_pool(nullptr)
{
    static constexpr std::array<int, 4> kSupportedSizes{32, 64, 128, 256};
    if (std::ranges::find(kSupportedSizes, n) == kSupportedSizes.end()) {
        throw std::runtime_error("GpuDct only supports sizes: 32, 64, 128, 256");
    }

    // Create stream if not provided
    if (m_owns_stream) {
        throw_on_cuda_error(cudaStreamCreate(&m_stream), "Failed to create CUDA stream");
    }

    // Create memory pool
    try {
        m_pool = std::make_unique<StreamMemoryPool>();
    } catch (...) {
        if (m_owns_stream) {
            cudaStreamDestroy(m_stream);
        }
        throw;
    }

    // Initialize transform matrix (ensures transform ready before first kernel)
    try {
        init_transform_matrix();
    } catch (...) {
        if (m_transform_device) {
            cudaFreeAsync(m_transform_device, m_stream);
            m_transform_device = nullptr;
        }
        m_transform_ptr = nullptr;
        m_transform_elements = 0;
        m_pool.reset();
        if (m_owns_stream && m_stream) {
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
        }
        throw;
    }
}

template<typename T>
    requires DctScalar<T>
GpuDct<T>::~GpuDct() {
    if (m_transform_device) {
        cudaFreeAsync(m_transform_device, m_stream);
        m_transform_device = nullptr;
    }

    if (m_owns_stream && m_stream) {
        cudaStreamSynchronize(m_stream);
        cudaStreamDestroy(m_stream);
    }
}

template<typename T>
    requires DctScalar<T>
GpuDct<T>::GpuDct(GpuDct&& other) noexcept
    : m_size(other.m_size)
    , m_stream(other.m_stream)
    , m_owns_stream(other.m_owns_stream)
    , m_pool(std::move(other.m_pool))
    , m_transform_device(std::exchange(other.m_transform_device, nullptr))
    , m_transform_ptr(other.m_transform_ptr)
    , m_transform_elements(other.m_transform_elements)
{
    other.m_stream = nullptr;
    other.m_owns_stream = false;
    other.m_transform_ptr = nullptr;
    other.m_transform_elements = 0;
}

template<typename T>
    requires DctScalar<T>
GpuDct<T>& GpuDct<T>::operator=(GpuDct&& other) noexcept {
    if (this != &other) {
        if (m_transform_device) {
            cudaFreeAsync(m_transform_device, m_stream);
        }
    if (m_owns_stream && m_stream) {
        cudaStreamSynchronize(m_stream);
        cudaStreamDestroy(m_stream);
    }
        m_size = other.m_size;
        m_stream = other.m_stream;
        m_owns_stream = other.m_owns_stream;
        m_pool = std::move(other.m_pool);
        m_transform_device = std::exchange(other.m_transform_device, nullptr);
        m_transform_ptr = other.m_transform_ptr;
        m_transform_elements = other.m_transform_elements;

        other.m_stream = nullptr;
        other.m_owns_stream = false;
        other.m_transform_ptr = nullptr;
        other.m_transform_elements = 0;
    }
    return *this;
}

// ============================================================================
// Transform Matrix Initialization
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::init_transform_matrix() {
    const int n = m_size;
    m_transform_elements = static_cast<size_t>(n) * static_cast<size_t>(n);

    auto set_constant_ptr = [&](const ComputeType* symbol_ptr) {
        m_transform_device = nullptr;
        m_transform_ptr = symbol_ptr;
    };

    if constexpr (std::is_same_v<ComputeType, float>) {
        if (n == 32) {
            static std::once_flag flag;
            std::call_once(flag, [] {
                auto host = make_dct_transform_array<ComputeType, 32>();
                throw_on_cuda_error(
                    cudaMemcpyToSymbol(c_dct_transform_f32_32, host.data(),
                                       host.size() * sizeof(ComputeType)),
                    "Failed to upload DCT transform to constant memory");
            });
            set_constant_ptr(reinterpret_cast<const ComputeType*>(c_dct_transform_f32_32));
            return;
        }
        if (n == 64) {
            static std::once_flag flag;
            std::call_once(flag, [] {
                auto host = make_dct_transform_array<ComputeType, 64>();
                throw_on_cuda_error(
                    cudaMemcpyToSymbol(c_dct_transform_f32_64, host.data(),
                                       host.size() * sizeof(ComputeType)),
                    "Failed to upload DCT transform to constant memory");
            });
            set_constant_ptr(reinterpret_cast<const ComputeType*>(c_dct_transform_f32_64));
            return;
        }
    }

    if constexpr (std::is_same_v<ComputeType, double>) {
        if (n == 32) {
            static std::once_flag flag;
            std::call_once(flag, [] {
                auto host = make_dct_transform_array<ComputeType, 32>();
                throw_on_cuda_error(
                    cudaMemcpyToSymbol(c_dct_transform_f64_32, host.data(),
                                       host.size() * sizeof(ComputeType)),
                    "Failed to upload DCT transform to constant memory");
            });
            set_constant_ptr(reinterpret_cast<const ComputeType*>(c_dct_transform_f64_32));
            return;
        }
        if (n == 64) {
            static std::once_flag flag;
            std::call_once(flag, [] {
                auto host = make_dct_transform_array<ComputeType, 64>();
                throw_on_cuda_error(
                    cudaMemcpyToSymbol(c_dct_transform_f64_64, host.data(),
                                       host.size() * sizeof(ComputeType)),
                    "Failed to upload DCT transform to constant memory");
            });
            set_constant_ptr(reinterpret_cast<const ComputeType*>(c_dct_transform_f64_64));
            return;
        }
    }

    // Fallback to device memory for larger matrices or unsupported constant storage
    auto host_transform = make_dct_transform_vector<ComputeType>(n);

    throw_on_cuda_error(
        cudaMallocAsync(reinterpret_cast<void**>(&m_transform_device),
                        m_transform_elements * sizeof(ComputeType),
                        m_stream),
        "Failed to allocate transform matrix on device");

    throw_on_cuda_error(
        cudaMemcpyAsync(m_transform_device, host_transform.data(),
                         m_transform_elements * sizeof(ComputeType),
                         cudaMemcpyHostToDevice, m_stream),
        "Failed to upload transform matrix");

    // Ensure transform ready for external streams
    throw_on_cuda_error(cudaStreamSynchronize(m_stream),
                        "Failed to synchronize after transform upload");

    m_transform_ptr = m_transform_device;
}

// ============================================================================
// Type Conversion
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::convert_to_compute_type(const T* d_input, ComputeType* d_output,
                                        size_t count, cudaStream_t stream) {
    if (std::is_same<T, ComputeType>::value) {
        // Same type, no conversion needed - cast both to void* for comparison
        if (static_cast<const void*>(d_input) != static_cast<const void*>(d_output)) {
            throw_on_cuda_error(
                cudaMemcpyAsync(d_output, d_input, count * sizeof(T),
                                 cudaMemcpyDeviceToDevice, stream),
                "Failed to copy device data");
        }
    } else {
        // Need conversion
        constexpr int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        kernel_convert_type<<<blocks, threads, 0, stream>>>(d_input, d_output,
                                                           static_cast<int>(count));
        throw_on_cuda_error(cudaGetLastError(),
                            "Failed to launch type conversion kernel");
    }
}

// ============================================================================
// Single Image DCT - Host Input
// ============================================================================

template<typename T>
    requires DctScalar<T>
uint64_t GpuDct<T>::dct_host(std::span<const T> image, cudaStream_t stream) {
    if (image.size() != static_cast<size_t>(m_size) * static_cast<size_t>(m_size)) {
        throw std::invalid_argument("dct_host: image span does not match configured size");
    }

    if (stream == nullptr) stream = m_stream;

    const size_t count = static_cast<size_t>(m_size) * static_cast<size_t>(m_size);

    // Allocate stream-ordered memory
    StreamMemory<T> d_input(count, stream, m_pool->get());
    StreamMemory<ComputeType> d_compute(count, stream, m_pool->get());
    StreamMemory<ComputeType> d_dct_output(count, stream, m_pool->get());
    StreamMemory<uint64_t> d_hash(1, stream, m_pool->get());

    // Upload image
    throw_on_cuda_error(
        cudaMemcpyAsync(d_input.get(), image.data(), count * sizeof(T),
                         cudaMemcpyHostToDevice, stream),
        "Failed to copy host image to device");

    // Convert to compute type if needed
    convert_to_compute_type(d_input.get(), d_compute.get(), count, stream);

    // Launch fused DCT kernel
    const ComputeType* transform = transform_ptr();

    if (m_size == 32) {
        kernel_fused_dct_32x32_warp<<<1, 1024, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    } else if (m_size == 64) {
        dim3 grid(1, 2, 2);
        dim3 block(32, 32);
        kernel_fused_dct_64x64<<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    } else if (m_size == 128) {
        constexpr int tile = 16;
        constexpr int tiles = (128 + tile - 1) / tile;
        dim3 grid(1, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 128, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    } else { // 256
        constexpr int tile = 16;
        constexpr int tiles = (256 + tile - 1) / tile;
        dim3 grid(1, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 256, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    }

    throw_on_cuda_error(cudaGetLastError(), "Failed to launch DCT kernel");

    // Download result
    uint64_t h_hash{};
    throw_on_cuda_error(
        cudaMemcpyAsync(&h_hash, d_hash.get(), sizeof(uint64_t),
                         cudaMemcpyDeviceToHost, stream),
        "Failed to copy hash to host");
    throw_on_cuda_error(cudaStreamSynchronize(stream),
                        "Failed to synchronize dct_host stream");

    return h_hash;
}

// ============================================================================
// Single Image DCT - Device Input
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::dct_device(const T* d_image, uint64_t* d_hash_out, cudaStream_t stream) {
    if (stream == nullptr) stream = m_stream;

    const size_t count = static_cast<size_t>(m_size) * static_cast<size_t>(m_size);

    // Allocate stream-ordered memory
    StreamMemory<ComputeType> d_compute(count, stream, m_pool->get());
    StreamMemory<ComputeType> d_dct_output(count, stream, m_pool->get());

    // Convert to compute type if needed
    convert_to_compute_type(d_image, d_compute.get(), count, stream);

    // Launch fused DCT kernel
    const ComputeType* transform = transform_ptr();

    if (m_size == 32) {
        kernel_fused_dct_32x32_warp<<<1, 1024, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash_out);
    } else if (m_size == 64) {
        dim3 grid(1, 2, 2);
        dim3 block(32, 32);
        kernel_fused_dct_64x64<<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash_out);
    } else if (m_size == 128) {
        constexpr int tile = 16;
        constexpr int tiles = (128 + tile - 1) / tile;
        dim3 grid(1, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 128, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash_out);
    } else { // 256
        constexpr int tile = 16;
        constexpr int tiles = (256 + tile - 1) / tile;
        dim3 grid(1, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 256, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash_out);
    }
    throw_on_cuda_error(cudaGetLastError(), "Failed to launch DCT kernel");
}

// ============================================================================
// Async Host DCT
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::dct_host_async(std::span<const T> image, uint64_t* h_hash_out, cudaStream_t stream) {
    if (image.size() != static_cast<size_t>(m_size) * static_cast<size_t>(m_size)) {
        throw std::invalid_argument("dct_host_async: image span does not match configured size");
    }

    if (stream == nullptr) stream = m_stream;

    const size_t count = static_cast<size_t>(m_size) * static_cast<size_t>(m_size);

    // Allocate stream-ordered memory
    StreamMemory<T> d_input(count, stream, m_pool->get());
    StreamMemory<ComputeType> d_compute(count, stream, m_pool->get());
    StreamMemory<ComputeType> d_dct_output(count, stream, m_pool->get());
    StreamMemory<uint64_t> d_hash(1, stream, m_pool->get());

    throw_on_cuda_error(
        cudaMemcpyAsync(d_input.get(), image.data(), count * sizeof(T),
                         cudaMemcpyHostToDevice, stream),
        "Failed to copy host image to device");

    convert_to_compute_type(d_input.get(), d_compute.get(), count, stream);

    const ComputeType* transform = transform_ptr();

    if (m_size == 32) {
        kernel_fused_dct_32x32_warp<<<1, 1024, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    } else if (m_size == 64) {
        dim3 grid(1, 2, 2);
        dim3 block(32, 32);
        kernel_fused_dct_64x64<<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    } else if (m_size == 128) {
        constexpr int tile = 16;
        constexpr int tiles = (128 + tile - 1) / tile;
        dim3 grid(1, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 128, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    } else {
        constexpr int tile = 16;
        constexpr int tiles = (256 + tile - 1) / tile;
        dim3 grid(1, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 256, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, 1, d_hash.get());
    }

    throw_on_cuda_error(cudaGetLastError(), "Failed to launch DCT kernel");

    throw_on_cuda_error(
        cudaMemcpyAsync(h_hash_out, d_hash.get(), sizeof(uint64_t),
                         cudaMemcpyDeviceToHost, stream),
        "Failed to copy hash to host");
}

// ============================================================================
// Batch DCT - Host Input
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::batch_dct_host(std::span<const T> images, std::span<uint64_t> hashes,
                               cudaStream_t stream) {
    if (hashes.empty()) {
        return;
    }

    const size_t expected = static_cast<size_t>(m_size) * static_cast<size_t>(m_size) * hashes.size();
    if (images.size() != expected) {
        throw std::invalid_argument("batch_dct_host: image span does not match batch size");
    }

    if (stream == nullptr) stream = m_stream;

    const size_t element_count = static_cast<size_t>(m_size) * static_cast<size_t>(m_size) * hashes.size();

    // Allocate stream-ordered memory
    StreamMemory<T> d_input(element_count, stream, m_pool->get());
    StreamMemory<ComputeType> d_compute(element_count, stream, m_pool->get());
    StreamMemory<ComputeType> d_dct_output(element_count, stream, m_pool->get());
    StreamMemory<uint64_t> d_hashes(hashes.size(), stream, m_pool->get());

    throw_on_cuda_error(
        cudaMemcpyAsync(d_input.get(), images.data(), element_count * sizeof(T),
                         cudaMemcpyHostToDevice, stream),
        "Failed to copy batched images to device");

    convert_to_compute_type(d_input.get(), d_compute.get(), element_count, stream);

    const ComputeType* transform = transform_ptr();
    const int batch_size = static_cast<int>(hashes.size());

    if (m_size == 32) {
        kernel_fused_dct_32x32_warp<<<batch_size, 1024, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes.get());
    } else if (m_size == 64) {
        dim3 grid(batch_size, 2, 2);
        dim3 block(32, 32);
        kernel_fused_dct_64x64<<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes.get());
    } else if (m_size == 128) {
        constexpr int tile = 16;
        constexpr int tiles = (128 + tile - 1) / tile;
        dim3 grid(batch_size, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 128, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes.get());
    } else {
        constexpr int tile = 16;
        constexpr int tiles = (256 + tile - 1) / tile;
        dim3 grid(batch_size, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 256, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes.get());
    }

    throw_on_cuda_error(cudaGetLastError(), "Failed to launch batched DCT kernel");

    throw_on_cuda_error(
        cudaMemcpyAsync(hashes.data(), d_hashes.get(), hashes.size() * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost, stream),
        "Failed to copy batched hashes to host");

    throw_on_cuda_error(cudaStreamSynchronize(stream),
                        "Failed to synchronize batch_dct_host stream");
}

// ============================================================================
// Batch DCT - Device Input
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::batch_dct_device(const T* d_images, uint64_t* d_hashes, 
                                 int batch_size, cudaStream_t stream) {
    if (batch_size <= 0) {
        if (batch_size < 0) {
            throw std::invalid_argument("batch_dct_device: negative batch size");
        }
        return;
    }

    if (stream == nullptr) stream = m_stream;
    
    const int n = m_size;
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n) * batch_size;
    
    // Allocate stream-ordered memory
    StreamMemory<ComputeType> d_compute(count, stream, m_pool->get());
    StreamMemory<ComputeType> d_dct_output(count, stream, m_pool->get());
    
    // Convert to compute type if needed
    convert_to_compute_type(d_images, d_compute.get(), count, stream);
    
    // Launch batched fused DCT kernel
    const ComputeType* transform = transform_ptr();
    
    if (n == 32) {
        kernel_fused_dct_32x32_warp<<<batch_size, 1024, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes
        );
    } else if (n == 64) {
        dim3 grid(batch_size, 2, 2);
        dim3 block(32, 32);
        kernel_fused_dct_64x64<<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes
        );
    } else if (n == 128) {
        constexpr int tile = 16;
        constexpr int tiles = (128 + tile - 1) / tile;
        dim3 grid(batch_size, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 128, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes
        );
    } else { // 256
        constexpr int tile = 16;
        constexpr int tiles = (256 + tile - 1) / tile;
        dim3 grid(batch_size, tiles, tiles);
        dim3 block(tile, tile);
        kernel_fused_dct_general<ComputeType, 256, 16><<<grid, block, 0, stream>>>(
            d_compute.get(), d_dct_output.get(), transform, batch_size, d_hashes
        );
    }

    throw_on_cuda_error(cudaGetLastError(), "Failed to launch batched DCT kernel");
}

// ============================================================================
// Multi-Stream Batch DCT
// ============================================================================

template<typename T>
    requires DctScalar<T>
void GpuDct<T>::batch_dct_device_multistream(const T* d_images, uint64_t* d_hashes,
                                             int batch_size, std::span<cudaStream_t> streams) {
    if (batch_size <= 0) {
        if (batch_size < 0) {
            throw std::invalid_argument("batch_dct_device_multistream: negative batch size");
        }
        return;
    }

    if (streams.empty()) {
        throw std::invalid_argument("batch_dct_device_multistream: no streams provided");
    }

    const int images_per_stream = (batch_size + static_cast<int>(streams.size()) - 1)
                                  / static_cast<int>(streams.size());

    for (size_t s = 0; s < streams.size(); ++s) {
        const int start_idx = static_cast<int>(s) * images_per_stream;
        const int count = std::min(images_per_stream, batch_size - start_idx);

        if (count <= 0) break;

        batch_dct_device(d_images + static_cast<size_t>(start_idx) * m_size * m_size,
                         d_hashes + start_idx,
                         count,
                         streams[s]);
    }

    for (cudaStream_t stream : streams) {
        if (stream) {
            throw_on_cuda_error(cudaStreamSynchronize(stream),
                                "Failed to synchronize multi-stream batch");
        }
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class GpuDct<float>;
template class GpuDct<double>;
template class GpuDct<int>;
template class GpuDct<unsigned int>;
template class GpuDct<uint8_t>;
template class GpuDct<int8_t>;
template class GpuDct<uint16_t>;
template class GpuDct<int16_t>;

} // namespace gpu_dct