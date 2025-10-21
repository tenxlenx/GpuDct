#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gpu_dct.cuh>

namespace gpu_dct {

// ============================================================================
// Constant Memory Declarations
// ============================================================================

/**
 * @brief Precomputed DCT-II transform matrix in constant memory for 32x32 float matrices.
 * 
 * This matrix contains the Type-II DCT coefficients where T[i,j] = c(i) * cos(π * i * (2j + 1) / (2N))
 * with c(0) = sqrt(1/N) and c(i) = sqrt(2/N) for i > 0.
 * Constant memory provides fast cached access for all threads in a warp.
 * 
 * @note Size: 32 * 32 * sizeof(float) = 4KB
 */
__constant__ float c_dct_transform_f32_32[32 * 32];

/**
 * @brief Precomputed DCT-II transform matrix in constant memory for 64x64 float matrices.
 * 
 * This matrix contains the Type-II DCT coefficients for 64x64 transformations.
 * Used by the 64x64 optimized kernel for faster memory access.
 * 
 * @note Size: 64 * 64 * sizeof(float) = 16KB
 */
__constant__ float c_dct_transform_f32_64[64 * 64];

/**
 * @brief Precomputed DCT-II transform matrix in constant memory for 32x32 double matrices.
 * 
 * Double precision version of the 32x32 DCT transform matrix.
 * Provides higher accuracy for applications requiring double precision.
 * 
 * @note Size: 32 * 32 * sizeof(double) = 8KB
 */
__constant__ double c_dct_transform_f64_32[32 * 32];

/**
 * @brief Precomputed DCT-II transform matrix in constant memory for 64x64 double matrices.
 * 
 * Double precision version of the 64x64 DCT transform matrix.
 * 
 * @note Size: 64 * 64 * sizeof(double) = 32KB
 */
__constant__ double c_dct_transform_f64_64[64 * 64];

// ============================================================================
// Type Conversion Kernels
// ============================================================================

/**
 * @brief Convert input type to compute type (e.g., int -> float)
 * 
 * Performs element-wise type conversion from input array to output array.
 * Commonly used to convert integer pixel values to floating-point for DCT computation.
 * 
 * @tparam TIn Input data type (e.g., uint8_t, int, float)
 * @tparam TOut Output data type (e.g., float, double)
 * @param input Pointer to input array in device memory
 * @param output Pointer to output array in device memory
 * @param count Total number of elements to convert
 * 
 * @note Launch configuration: 1D grid with (count + block_size - 1) / block_size blocks
 * @note Thread safety: Each thread processes one element independently
 */
template <typename TIn, typename TOut>
__global__ void kernel_convert_type(const TIn *input, TOut *output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = static_cast<TOut>(input[idx]);
    }
}

// ============================================================================
// DCT Transform Matrix Generation
// ============================================================================

/**
 * @brief Generate DCT transform matrix (Type-II DCT)
 *
 * Computes the DCT-II basis functions: T[i,j] = c(i) * cos(π * i * (2j + 1) / (2N))
 * where c(0) = sqrt(1/N), c(i) = sqrt(2/N) for i > 0
 * 
 * The resulting matrix can be used to compute forward DCT via: DCT = T * A * T^T
 * where A is the input matrix and T is the transform matrix.
 * 
 * @tparam T Floating-point type (float or double)
 * @param output Pointer to output transform matrix in device memory [n x n]
 * @param n Dimension of the square transform matrix (e.g., 32, 64, 128, 256)
 * 
 * @note Launch configuration: Grid(n, 1, 1), Block(n, 1, 1)
 * @note Each thread computes one element of the transform matrix
 * @note blockIdx.x represents row index, threadIdx.x represents column index
 */
template <typename T>
__global__ void kernel_generate_dct_transform(T *output, int n) {
    int i = blockIdx.x;  // Row
    int j = threadIdx.x; // Column

    if (i >= n || j >= n)
        return;

    const T PI = static_cast<T>(3.14159265358979323846);
    const T c_i =
        (i == 0) ? sqrt(static_cast<T>(1) / n) : sqrt(static_cast<T>(2) / n);

    output[i * n + j] = c_i * cos(PI * i * (2 * j + 1) / (2 * n));
}

// ============================================================================
// Fused DCT Kernel - Optimized for 32x32 (Warp-Level)
// ============================================================================

/**
 * @brief Fused DCT kernel optimized for 32x32 matrices
 *
 * Computes: DCT = T * A * T^T in a single kernel pass
 * Uses 32 warps (1024 threads) where each warp computes one row
 * Exploits warp-level parallelism and shared memory
 * 
 * Algorithm:
 * 1. Load 32x32 input matrix into shared memory
 * 2. Compute T * A using warp-level parallelism (each warp = one row)
 * 3. Compute (T * A) * T^T to get final DCT
 * 4. Optionally compute perceptual hash from top-left 8x8 DCT coefficients
 * 
 * Hash computation:
 * - Extracts 8x8 top-left DCT coefficients
 * - Computes median using parallel odd-even sort
 * - Generates 64-bit hash where bit i is set if coefficient i > median
 * 
 * @tparam T Floating-point compute type (float or double)
 * @param input Input matrices in device memory [batch_size, 32, 32]
 * @param output Output DCT matrices in device memory [batch_size, 32, 32], can be nullptr if only hash is needed
 * @param transform DCT transform matrix in constant memory [32, 32]
 * @param batch_size Number of matrices to process
 * @param hashes Optional output perceptual hashes [batch_size], nullptr to skip hash computation
 * 
 * @note Launch configuration: Grid(batch_size, 1, 1), Block(1024, 1, 1)
 * @note Shared memory usage: 3 * 32 * 32 * sizeof(T) + 64 * 2 * sizeof(T) bytes
 * @note Optimized for warp-level operations, achieves high occupancy
 * 
 * Grid: (batch_size, 1, 1)
 * Block: (1024, 1, 1) = 32 warps of 32 threads
 */
template <typename T>
__global__ void kernel_fused_dct_32x32_warp(
    const T *__restrict__ input,     // [batch_size, 32, 32]
    T *__restrict__ output,          // [batch_size, 32, 32]
    const T *__restrict__ transform, // [32, 32] in constant memory
    int batch_size,
    uint64_t *__restrict__ hashes // [batch_size] (optional)
) {
    const int N = 32;
    const int batch_idx = blockIdx.x;

    if (batch_idx >= batch_size)
        return;

    // Each warp computes one row of the output
    const int warp_id = threadIdx.x / 32; // 0-31
    const int lane_id = threadIdx.x % 32; // 0-31

    // Shared memory for input matrix and intermediate result
    __shared__ T s_input[N * N];
    __shared__ T s_temp[N * N];
    __shared__ T s_hash_values[64];
    __shared__ T s_sorted[64];

    // Load input matrix to shared memory (coalesced)
    const T *input_base = input + batch_idx * N * N;
    for (int i = threadIdx.x; i < N * N; i += blockDim.x) {
        s_input[i] = input_base[i];
    }
    __syncthreads();

    // Step 1: Compute T * A -> s_temp
    // Each warp computes one row of T * A
    if (warp_id < N) {
        T sum = 0;
        for (int k = 0; k < N; k++) {
            sum += transform[warp_id * N + k] * s_input[k * N + lane_id];
        }
        s_temp[warp_id * N + lane_id] = sum;
    }
    __syncthreads();

    // Step 2: Compute (T * A) * T^T -> output
    // Each warp computes one row of (T * A) * T^T
    if (warp_id < N) {
        T sum = 0;
        for (int k = 0; k < N; k++) {
            sum += s_temp[warp_id * N + k] * transform[lane_id * N + k];
        }
        if (output) {
            output[batch_idx * N * N + warp_id * N + lane_id] = sum;
        }
        if (hashes && warp_id < 8 && lane_id < 8) {
            s_hash_values[warp_id * 8 + lane_id] = sum;
        }
    }

    __syncthreads();

    if (hashes) {
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        const bool active = (warp_id < 8) && (lane_id < 8);
        const int hash_lane = warp_id * 8 + lane_id;

        if (active) {
            s_sorted[hash_lane] = s_hash_values[hash_lane];
        }
        __syncthreads();

        for (int i = 0; i < 32; ++i) {
            if (active && hash_lane < 63 && (hash_lane % 2 == 0)) {
                if (s_sorted[hash_lane] > s_sorted[hash_lane + 1]) {
                    T temp = s_sorted[hash_lane];
                    s_sorted[hash_lane] = s_sorted[hash_lane + 1];
                    s_sorted[hash_lane + 1] = temp;
                }
            }
            __syncthreads();

            if (active && hash_lane < 63 && (hash_lane % 2 == 1)) {
                if (s_sorted[hash_lane] > s_sorted[hash_lane + 1]) {
                    T temp = s_sorted[hash_lane];
                    s_sorted[hash_lane] = s_sorted[hash_lane + 1];
                    s_sorted[hash_lane + 1] = temp;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const T median = (s_sorted[31] + s_sorted[32]) / static_cast<T>(2);
            uint64_t hash = 0ULL;
            for (int i = 0; i < 64; ++i) {
                if (s_hash_values[i] > median) {
                    hash |= (uint64_t{1} << i);
                }
            }
            hashes[batch_idx] = hash;
        }
    }
}

// ============================================================================
// Fused DCT Kernel - Optimized for 64x64
// ============================================================================

/**
 * @brief Fused DCT kernel optimized for 64x64 matrices
 *
 * Uses tiled computation with 32x32 tiles for efficient shared memory usage.
 * Processes the matrix multiplication in two stages with tiling to fit in shared memory.
 * 
 * Algorithm:
 * 1. Tile-based computation of T * A using 32x32 tiles
 * 2. Tile-based computation of (T * A) * T^T
 * 3. Optional perceptual hash from 8x8 top-left coefficients
 * 
 * Tiling strategy:
 * - Divides 64x64 matrix into 2x2 grid of 32x32 tiles
 * - Each block processes one tile of the result
 * - Uses shared memory to cache tiles and avoid redundant global memory access
 * 
 * @tparam T Floating-point compute type (float or double)
 * @param input Input matrices in device memory [batch_size, 64, 64]
 * @param output Output DCT matrices in device memory [batch_size, 64, 64]
 * @param transform DCT transform matrix in constant memory [64, 64]
 * @param batch_size Number of matrices to process
 * @param hashes Optional output perceptual hashes [batch_size], nullptr to skip
 * 
 * @note Launch configuration: Grid(batch_size, 2, 2), Block(32, 32, 1)
 * @note Shared memory usage: 3 * (32 * 33) * sizeof(T) + 128 * sizeof(T) bytes
 * @note Bank conflict avoidance: Uses padding (+1) in shared memory arrays
 * 
 * Grid: (batch_size, 2, 2)
 * Block: (32, 32, 1) = 1024 threads
 */
template <typename T>
__global__ void
kernel_fused_dct_64x64(const T *__restrict__ input, T *__restrict__ output,
                       const T *__restrict__ transform, int batch_size,
                       uint64_t *__restrict__ hashes) {
    const int N = 64;
    const int TILE = 32;

    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ T s_A[TILE][TILE + 1]; // +1 to avoid bank conflicts
    __shared__ T s_T[TILE][TILE + 1];
    __shared__ T s_temp[TILE][TILE + 1];
    __shared__ T s_hash_values[64];
    __shared__ T s_sorted[64];

    const T *input_base = input + batch_idx * N * N;
    T *output_base = output + batch_idx * N * N;

    T acc = 0;

    // Compute T * A (tiled)
    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Load tiles
        int row = tile_row * TILE + ty;
        int col = tile * TILE + tx;
        s_T[ty][tx] = (row < N && col < N) ? transform[row * N + col] : 0;

        row = tile * TILE + ty;
        col = tile_col * TILE + tx;
        s_A[ty][tx] = (row < N && col < N) ? input_base[row * N + col] : 0;

        __syncthreads();

        // Compute partial result
        for (int k = 0; k < TILE; k++) {
            acc += s_T[ty][k] * s_A[k][tx];
        }
        __syncthreads();
    }

    s_temp[ty][tx] = acc;
    __syncthreads();

    // Compute (T * A) * T^T
    acc = 0;
    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Load transform tile for transpose
        int row = tile * TILE + ty;
        int col = tile_col * TILE + tx;
        s_T[ty][tx] = (row < N && col < N) ? transform[col * N + row] : 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc += s_temp[ty][k] * s_T[tx][k];
        }
        __syncthreads();
    }

    // Write output
    int out_row = tile_row * TILE + ty;
    int out_col = tile_col * TILE + tx;
    if (out_row < N && out_col < N) {
        output_base[out_row * N + out_col] = acc;
        if (hashes && tile_row == 0 && tile_col == 0 && ty < 8 && tx < 8) {
            s_hash_values[ty * 8 + tx] = acc;
        }
    }

    __syncthreads();

    if (hashes && tile_row == 0 && tile_col == 0) {
        const bool active = (ty < 8) && (tx < 8);
        const int hash_lane = ty * 8 + tx;

        if (active) {
            s_sorted[hash_lane] = s_hash_values[hash_lane];
        }
        __syncthreads();

        for (int i = 0; i < 32; ++i) {
            if (active && hash_lane < 63 && (hash_lane % 2 == 0)) {
                if (s_sorted[hash_lane] > s_sorted[hash_lane + 1]) {
                    T temp = s_sorted[hash_lane];
                    s_sorted[hash_lane] = s_sorted[hash_lane + 1];
                    s_sorted[hash_lane + 1] = temp;
                }
            }
            __syncthreads();

            if (active && hash_lane < 63 && (hash_lane % 2 == 1)) {
                if (s_sorted[hash_lane] > s_sorted[hash_lane + 1]) {
                    T temp = s_sorted[hash_lane];
                    s_sorted[hash_lane] = s_sorted[hash_lane + 1];
                    s_sorted[hash_lane + 1] = temp;
                }
            }
            __syncthreads();
        }

        if (ty == 0 && tx == 0) {
            const T median = (s_sorted[31] + s_sorted[32]) / static_cast<T>(2);
            uint64_t hash = 0ULL;
            for (int i = 0; i < 64; ++i) {
                if (s_hash_values[i] > median) {
                    hash |= (uint64_t{1} << i);
                }
            }
            hashes[batch_idx] = hash;
        }
    }
}

// ============================================================================
// General Fused DCT Kernel (128x128, 256x256)
// ============================================================================

/**
 * @brief General fused DCT kernel for larger matrices
 *
 * Uses tiled matrix multiplication with configurable tile size.
 * Performs computation in two passes, storing intermediate results in global memory.
 * Suitable for matrices larger than 64x64 (e.g., 128x128, 256x256).
 * 
 * Algorithm:
 * 1. First pass: Compute T * A with tiled multiplication, store in output buffer
 * 2. __syncthreads() to ensure intermediate results are visible
 * 3. Second pass: Compute (T * A) * T^T with tiled multiplication
 * 4. Optional hash computation from 8x8 top-left coefficients
 * 
 * Memory access pattern:
 * - Uses shared memory for tiles to maximize cache reuse
 * - Coalesced global memory access for input/output
 * - Bank conflict avoidance with shared memory padding
 * 
 * @tparam T Floating-point compute type (float or double)
 * @tparam N Matrix dimension (128, 256, or other power of 2)
 * @tparam TILE Tile size for shared memory (typically 32)
 * @param input Input matrices in device memory [batch_size, N, N]
 * @param output Output DCT matrices in device memory [batch_size, N, N]
 * @param transform DCT transform matrix in global memory [N, N]
 * @param batch_size Number of matrices to process
 * @param hashes Optional output perceptual hashes [batch_size], nullptr to skip
 * 
 * @note Launch configuration: Grid(batch_size, ceil(N/TILE), ceil(N/TILE)), Block(TILE, TILE, 1)
 * @note Shared memory usage: 2 * (TILE * (TILE+1)) * sizeof(T) + 128 * sizeof(T) bytes
 * @note Requires intermediate storage in global memory between two passes
 * @note For N > 256, consider using cuBLAS for better performance
 * 
 * Grid: (batch_size, ceil(N/32), ceil(N/32))
 * Block: (32, 32, 1)
 */
template <typename T, int N, int TILE>
__global__ void
kernel_fused_dct_general(const T *__restrict__ input, T *__restrict__ output,
                         const T *__restrict__ transform, int batch_size,
                         uint64_t *__restrict__ hashes) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ T s_A[TILE][TILE + 1];
    __shared__ T s_T[TILE][TILE + 1];
    __shared__ T s_hash_values[64];
    __shared__ T s_sorted[64];

    const T *input_base = input + batch_idx * N * N;
    T *output_base = output + batch_idx * N * N;

    // Step 1: Compute T * A -> temp (stored in global memory)
    T acc1 = 0;

    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Load tiles
        int row = tile_row * TILE + ty;
        int col = tile * TILE + tx;
        s_T[ty][tx] = (row < N && col < N) ? transform[row * N + col] : 0;

        row = tile * TILE + ty;
        col = tile_col * TILE + tx;
        s_A[ty][tx] = (row < N && col < N) ? input_base[row * N + col] : 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc1 += s_T[ty][k] * s_A[k][tx];
        }
        __syncthreads();
    }

    // Store intermediate result
    int out_row = tile_row * TILE + ty;
    int out_col = tile_col * TILE + tx;
    if (out_row < N && out_col < N) {
        output_base[out_row * N + out_col] = acc1;
    }
    __syncthreads();

    // Step 2: Compute (T * A) * T^T
    T acc2 = 0;

    for (int tile = 0; tile < (N + TILE - 1) / TILE; tile++) {
        // Load tiles
        int row = tile_row * TILE + ty;
        int col = tile * TILE + tx;
        s_A[ty][tx] = (row < N && col < N) ? output_base[row * N + col] : 0;

        row = tile_col * TILE + ty;
        col = tile * TILE + tx;
        s_T[ty][tx] = (row < N && col < N) ? transform[row * N + col] : 0;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            acc2 += s_A[ty][k] * s_T[tx][k];
        }
        __syncthreads();
    }

    if (out_row < N && out_col < N) {
        output_base[out_row * N + out_col] = acc2;
        if (hashes && tile_row == 0 && tile_col == 0 && ty < 8 && tx < 8) {
            s_hash_values[ty * 8 + tx] = acc2;
        }
    }

    __syncthreads();

    if (hashes && tile_row == 0 && tile_col == 0) {
        const bool active = (ty < 8) && (tx < 8);
        const int hash_lane = ty * 8 + tx;

        if (active) {
            s_sorted[hash_lane] = s_hash_values[hash_lane];
        }
        __syncthreads();

        for (int i = 0; i < 32; ++i) {
            if (active && hash_lane < 63 && (hash_lane % 2 == 0)) {
                if (s_sorted[hash_lane] > s_sorted[hash_lane + 1]) {
                    T temp = s_sorted[hash_lane];
                    s_sorted[hash_lane] = s_sorted[hash_lane + 1];
                    s_sorted[hash_lane + 1] = temp;
                }
            }
            __syncthreads();

            if (active && hash_lane < 63 && (hash_lane % 2 == 1)) {
                if (s_sorted[hash_lane] > s_sorted[hash_lane + 1]) {
                    T temp = s_sorted[hash_lane];
                    s_sorted[hash_lane] = s_sorted[hash_lane + 1];
                    s_sorted[hash_lane + 1] = temp;
                }
            }
            __syncthreads();
        }

        if (ty == 0 && tx == 0) {
            const T median = (s_sorted[31] + s_sorted[32]) / static_cast<T>(2);
            uint64_t hash = 0ULL;
            for (int i = 0; i < 64; ++i) {
                if (s_hash_values[i] > median) {
                    hash |= (uint64_t{1} << i);
                }
            }
            hashes[batch_idx] = hash;
        }
    }
}

} // namespace gpu_dct
