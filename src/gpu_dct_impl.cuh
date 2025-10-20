#pragma once

#include "../include/gpu_dct.cuh"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace gpu_dct {

// ============================================================================
// Constant Memory Declarations
// ============================================================================

__constant__ float c_dct_transform_f32_32[32 * 32];
__constant__ float c_dct_transform_f32_64[64 * 64];
__constant__ double c_dct_transform_f64_32[32 * 32];
__constant__ double c_dct_transform_f64_64[64 * 64];

// ============================================================================
// Type Conversion Kernels
// ============================================================================

/**
 * @brief Convert input type to compute type (e.g., int -> float)
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
 * T[i,j] = c(i) * cos(π * i * (2j + 1) / (2N))
 * where c(0) = sqrt(1/N), c(i) = sqrt(2/N) for i > 0
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
 * Uses tiled computation with 32x32 tiles
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
 * Uses tiled matrix multiplication
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
