#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// Namespace for GPU DCT library
namespace gpu_dct {

// Constants for DCT transform matrices in constant memory
// Optimized for common image sizes: 32x32, 64x64, 128x128, 256x256
__constant__ extern float c_dct_transform_f32_32[32 * 32];
__constant__ extern float c_dct_transform_f32_64[64 * 64];
__constant__ extern double c_dct_transform_f64_32[32 * 32];
__constant__ extern double c_dct_transform_f64_64[64 * 64];

// Forward declarations
class StreamMemoryPool;
template <typename T> class StreamMemory;

// ============================================================================
// Utility helpers
// ============================================================================

void throw_on_cuda_error(cudaError_t error, std::string_view context);

template <typename T>
concept DctScalar = std::is_arithmetic_v<T>;

/**
 * @brief RAII wrapper for stream-ordered memory allocation
 *
 * Automatically frees memory when object goes out of scope.
 * Memory lifetime is tied to CUDA stream ordering.
 *
 * @tparam T Data type
 */
template <typename T> class StreamMemory {
  public:
    StreamMemory(size_t count, cudaStream_t stream,
                 cudaMemPool_t pool = nullptr)
        : m_ptr(nullptr), m_stream(stream), m_count(count) {

        if (count == 0)
            return;

        cudaError_t err;
        if (pool) {
            err = cudaMallocFromPoolAsync(&m_ptr, count * sizeof(T), pool,
                                          stream);
        } else {
            err = cudaMallocAsync(&m_ptr, count * sizeof(T), stream);
        }

        if (err != cudaSuccess) {
            throw std::runtime_error("StreamMemory allocation failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

    ~StreamMemory() {
        if (m_ptr) {
            cudaFreeAsync(m_ptr, m_stream);
        }
    }

    // Non-copyable
    StreamMemory(const StreamMemory &) = delete;
    StreamMemory &operator=(const StreamMemory &) = delete;

    // Movable
    StreamMemory(StreamMemory &&other) noexcept
        : m_ptr(std::exchange(other.m_ptr, nullptr)),
          m_stream(std::exchange(other.m_stream, cudaStream_t{})),
          m_count(std::exchange(other.m_count, size_t{0})) {}

    StreamMemory &operator=(StreamMemory &&other) noexcept {
        if (this != &other) {
            if (m_ptr) {
                cudaFreeAsync(m_ptr, m_stream);
            }
            m_ptr = std::exchange(other.m_ptr, nullptr);
            m_stream = std::exchange(other.m_stream, cudaStream_t{});
            m_count = std::exchange(other.m_count, size_t{0});
        }
        return *this;
    }

    [[nodiscard]] T *get() { return static_cast<T *>(m_ptr); }
    [[nodiscard]] const T *get() const { return static_cast<const T *>(m_ptr); }
    [[nodiscard]] size_t count() const { return m_count; }

  private:
    void *m_ptr;
    cudaStream_t m_stream;
    size_t m_count;
};

/**
 * @brief Stream-ordered memory pool for efficient allocation reuse
 */
class StreamMemoryPool {
  public:
    StreamMemoryPool() : m_pool(nullptr) {
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        cudaGetDevice(&props.location.id);

        throw_on_cuda_error(cudaMemPoolCreate(&m_pool, &props),
                            "Failed to create memory pool");

        // Set release threshold to maximum to keep memory allocated
        uint64_t threshold = UINT64_MAX;
        throw_on_cuda_error(
            cudaMemPoolSetAttribute(m_pool, cudaMemPoolAttrReleaseThreshold,
                                    &threshold),
            "Failed to configure memory pool");
    }

    ~StreamMemoryPool() {
        if (m_pool) {
            cudaMemPoolDestroy(m_pool);
        }
    }

    // Non-copyable
    StreamMemoryPool(const StreamMemoryPool &) = delete;
    StreamMemoryPool &operator=(const StreamMemoryPool &) = delete;

    [[nodiscard]] cudaMemPool_t get() const { return m_pool; }

  private:
    cudaMemPool_t m_pool;
};

/**
 * @brief Main GPU DCT class - fully templated for various data types
 *
 * Supports float, double, int, uint8_t, etc. Automatically converts integral
 * types to floating point for DCT computation. Optimized for image sizes:
 * 32x32, 64x64, 128x128, 256x256
 *
 * Features:
 * - Stream-ordered memory allocation (zero overhead)
 * - Fused DCT kernel (single pass TAT' computation)
 * - Constant memory for transform matrices
 * - Batch processing with multi-stream support
 * - Type-safe templated interface
 *
 * @tparam T Input data type (float, double, int, uint8_t, etc.)
 */
template <typename T>
    requires DctScalar<T>
class GpuDct {
  public:
    // Compute type: float for integral types, same as T for floating types
    using ComputeType =
        typename std::conditional<std::is_integral<T>::value, float, T>::type;

    /**
     * @brief Construct a new GpuDct object
     *
     * @param n Image dimension (must be 32, 64, 128, or 256)
     * @param stream CUDA stream to use (nullptr for default stream)
     */
    explicit GpuDct(int n, cudaStream_t stream = nullptr);

    /**
     * @brief Destroy the GpuDct object
     *
     * Automatically cleans up stream-ordered memory
     */
    ~GpuDct();

    // Non-copyable
    GpuDct(const GpuDct &) = delete;
    GpuDct &operator=(const GpuDct &) = delete;

    // Movable
    GpuDct(GpuDct &&) noexcept;
    GpuDct &operator=(GpuDct &&) noexcept;

    /**
     * @brief Compute DCT hash from host memory
     *
     * @param h_image Host image data (row-major, size n*n)
     * @param stream CUDA stream (nullptr = use default from constructor)
     * @return Hash value
     */
    [[nodiscard]] uint64_t dct_host(std::span<const T> image,
                                    cudaStream_t stream = nullptr);
    [[nodiscard]] uint64_t dct_host(const T *h_image,
                                    cudaStream_t stream = nullptr) {
        if (h_image == nullptr && m_size > 0) {
            throw std::invalid_argument("dct_host: h_image pointer is null");
        }
        return dct_host(
            std::span<const T>{h_image, static_cast<size_t>(m_size * m_size)},
            stream);
    }

    /**
     * @brief Compute DCT hash from device memory
     *
     * @param d_image Device image data (row-major, size n*n)
     * @param d_hash_out Device hash output (allocated by caller)
     * @param stream CUDA stream (nullptr = use default from constructor)
     */
    void dct_device(const T *d_image, uint64_t *d_hash_out,
                    cudaStream_t stream = nullptr);

    /**
     * @brief Compute DCT hash from host memory (async)
     *
     * Requires pinned host memory for optimal performance.
     * Caller must synchronize stream before reading result.
     *
     * @param h_image Host image data (pinned)
     * @param h_hash_out Host hash output (pinned)
     * @param stream CUDA stream (nullptr = use default from constructor)
     */
    void dct_host_async(std::span<const T> image, uint64_t *h_hash_out,
                        cudaStream_t stream = nullptr);
    void dct_host_async(const T *h_image, uint64_t *h_hash_out,
                        cudaStream_t stream = nullptr) {
        if ((h_image == nullptr || h_hash_out == nullptr) && m_size > 0) {
            throw std::invalid_argument("dct_host_async: null host pointer");
        }
        dct_host_async(
            std::span<const T>{h_image, static_cast<size_t>(m_size * m_size)},
            h_hash_out, stream);
    }

    /**
     * @brief Batch compute DCT hashes from host memory
     *
     * @param h_images Host image data (row-major, size n*n*batch_size)
     * @param h_hashes Host hash outputs (size batch_size)
     * @param batch_size Number of images
     * @param stream CUDA stream (nullptr = use default from constructor)
     */
    void batch_dct_host(std::span<const T> images, std::span<uint64_t> hashes,
                        cudaStream_t stream = nullptr);
    void batch_dct_host(const T *h_images, uint64_t *h_hashes, int batch_size,
                        cudaStream_t stream = nullptr) {
        if (batch_size < 0) {
            throw std::invalid_argument("batch_dct_host: negative batch size");
        }
        if ((batch_size > 0) && (h_images == nullptr || h_hashes == nullptr)) {
            throw std::invalid_argument("batch_dct_host: null host pointer");
        }
        batch_dct_host(
            std::span<const T>{
                h_images, static_cast<size_t>(m_size * m_size * batch_size)},
            std::span<uint64_t>{h_hashes, static_cast<size_t>(batch_size)},
            stream);
    }

    /**
     * @brief Batch compute DCT hashes from device memory
     *
     * @param d_images Device image data (row-major, size n*n*batch_size)
     * @param d_hashes Device hash outputs (size batch_size)
     * @param batch_size Number of images
     * @param stream CUDA stream (nullptr = use default from constructor)
     */
    void batch_dct_device(const T *d_images, uint64_t *d_hashes, int batch_size,
                          cudaStream_t stream = nullptr);

    /**
     * @brief Batch compute with multi-stream parallelism
     *
     * Divides batch across multiple streams for maximum throughput.
     *
     * @param d_images Device image data (row-major, size n*n*batch_size)
     * @param d_hashes Device hash outputs (size batch_size)
     * @param batch_size Number of images
     * @param streams Array of CUDA streams
     * @param num_streams Number of streams
     */
    void batch_dct_device_multistream(const T *d_images, uint64_t *d_hashes,
                                      int batch_size,
                                      std::span<cudaStream_t> streams);
    void batch_dct_device_multistream(const T *d_images, uint64_t *d_hashes,
                                      int batch_size, cudaStream_t *streams,
                                      int num_streams) {
        if (batch_size < 0) {
            throw std::invalid_argument(
                "batch_dct_device_multistream: negative batch size");
        }
        if ((batch_size > 0) && (d_images == nullptr || d_hashes == nullptr)) {
            throw std::invalid_argument(
                "batch_dct_device_multistream: null device pointer");
        }
        if (num_streams < 0) {
            throw std::invalid_argument(
                "batch_dct_device_multistream: negative num_streams");
        }
        if (num_streams > 0 && streams == nullptr) {
            throw std::invalid_argument(
                "batch_dct_device_multistream: null stream array");
        }
        batch_dct_device_multistream(
            d_images, d_hashes, batch_size,
            std::span<cudaStream_t>{streams, static_cast<size_t>(num_streams)});
    }

    /**
     * @brief Get image dimension
     */
    int size() const { return m_size; }

    /**
     * @brief Get default stream
     */
    cudaStream_t stream() const { return m_stream; }

  private:
    int m_size;                               // Image dimension (n)
    cudaStream_t m_stream;                    // Default CUDA stream
    bool m_owns_stream;                       // Whether we created the stream
    std::unique_ptr<StreamMemoryPool> m_pool; // Memory pool for allocations

    // Transform data management
    ComputeType *m_transform_device{nullptr};
    const ComputeType *m_transform_ptr{nullptr};
    size_t m_transform_elements{0};

    // Initialize DCT transform matrix
    void init_transform_matrix();

    [[nodiscard]] const ComputeType *transform_ptr() const {
        return m_transform_ptr;
    }

    // Convert input data to compute type if needed
    void convert_to_compute_type(const T *d_input, ComputeType *d_output,
                                 size_t count, cudaStream_t stream);
}; // class GpuDct

} // namespace gpu_dct
