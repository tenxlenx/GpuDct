#include "../include/gpu_dct.cuh"

#include "device_launch_parameters.h"


#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cuda.h"

/**
 * @brief adds ap all the bit values in a warp
 *
 * @param s_data shared memory containing the values we need to add up to get the final value
 * @param tid thread id
 *
 */
__device__ void warp_add_bits(volatile unsigned long long int* s_data, int tid)
{
    if (tid < 32) {
        s_data[tid] += s_data[tid + 32];
        s_data[tid] += s_data[tid + 16];
        s_data[tid] += s_data[tid + 8];
        s_data[tid] += s_data[tid + 4];
        s_data[tid] += s_data[tid + 2];
        s_data[tid] += s_data[tid + 1];
    }
}

/**
 * @brief this function adds up the bits in a bit array to create an unsigned long long integer
 * number of blocks should be the number of elements to make hash values
 * and the number of threads should be 64 - to be consistent with std::bitset implementation
 * this version does not reverse the bit sequence
 *
 * @param bit_array array of int values containing 64 x num elements
 * @param binary array of unsigned long long int values x num elements
 *
 */
__global__ void kernel_add_bits(const int* bit_array, unsigned long long int* binary)
{
    int uid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ unsigned long long int reversed_bit_values[64];
    __shared__ unsigned long long int bit_values[64];
    __shared__ int s_bit_array[64];
    __shared__ int reversed_bit_array[64];

    // load bit values to shared memory
    s_bit_array[tid] = bit_array[uid];
    __syncthreads();

    double x = __int2double_rn(s_bit_array[tid]);
    bit_values[tid] = __double2ull_rn(scalbn(x, tid));
    __syncthreads();

    // add up all the numbers to make the bit values
    warp_add_bits(bit_values, tid);
    __syncthreads();

    // store result
    binary[bid] = bit_values[0];
    __syncthreads();
}

__global__ void kernel_get_T(float* T)
{
    int uid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    float j = (float)tid;
    float i = (float)bid;
    float N = (float)blockDim.x;
    const float PI = 3.141592654;
    // different normalisation for first column

    if (bid == 0) {
        T[uid] = sqrt(1.0 / N) * cos(((2 * j + 1)) / (2.0 * N) * PI * i);
    }
    else {
        T[uid] = sqrt(2.0 / N) * cos(((2 * j + 1)) / (2.0 * N) * PI * i);
    }
}

//! threads per block = 64 x n_images, blocks = n_images
__global__ void kernel_compute_hash(float* dct_imgs, int* bit_array, int cols)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    int roi_offset = tid % 8 + ((tid / 8) * cols);
    __shared__ float s_array[64];
    __shared__ float sb_array[64];
    s_array[threadIdx.x] = dct_imgs[roi_offset + bid * blockIdx.x];
    sb_array[threadIdx.x] = dct_imgs[roi_offset + bid * blockIdx.x];
    __syncthreads();

    // sort arrays
    for (int i = 0; i < cols / 2; i++) {
        int j = tid;
        if (j % 2 == 0 && j < cols - 1) {
            if (s_array[j + 1] < s_array[j]) {
                float temp = s_array[j];
                s_array[j] = s_array[j + 1];
                s_array[j + 1] = temp;
            }
        }
        __syncthreads();
        if (j % 2 == 1 && j < cols - 1) {
            if (s_array[j + 1] < s_array[j]) {
                float temp = s_array[j];
                s_array[j] = s_array[j + 1];
                s_array[j + 1] = temp;
            }
        }
        __syncthreads();
    }
    __syncthreads();

    const float median = (s_array[31] + s_array[32]) / 2;
    if (sb_array[tid] > median) {
        bit_array[tid] = 1;
    }
    else {
        bit_array[tid] = 0;
    }
    __syncthreads();
}

void GpuDct::setHandle(const cublasHandle_t& _handle)
{
    m_handle = _handle;
}

GpuDct::GpuDct(int n)
{
    m_size = n;
    cudaMalloc(&d_T, n * n * sizeof(float));
    getDCTMatrix(n, n, d_T);
    cublasCreate(&m_handle);
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_bits, 64 * sizeof(int));
    cudaMalloc(&d_tmp, n * n * sizeof(float));
    cudaMalloc(&d_DCT, n * n * sizeof(float));
}

GpuDct::~GpuDct()
{
    cublasDestroy(m_handle);
    cudaFree(d_T);
    cudaFree(d_bits);
    cudaFree(d_tmp);
    cudaFree(d_A);
    cudaFree(d_DCT);
}

/**
 * @brief calculate the dct hash value of a single image and return the hash value
 *
 * @param img image
 * @return hash value
 */
unsigned long long int GpuDct::dct(const cv::Mat& img)
{
    unsigned long long int* d_hash_val;
    cudaMalloc(&d_hash_val, sizeof(unsigned long long int));

    cudaMemset(d_A, 0, m_size * m_size * sizeof(float));
    cudaMemcpy(d_A, reinterpret_cast<float*>(img.data), m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);
    unsigned long long int hash = gpu_dct_single(d_A, d_hash_val);
    cudaFree(d_hash_val);
    return hash;
}

/**
 * @brief calculates dct hash
 *
 * @param d_img GPU image
 * @param d_hash GPU hash value hash gets copied to user supplied hash value - need to allocate memory on calling side
 *
 */
void GpuDct::dct(const float* d_img, unsigned long long int* d_hash_val)
{
    unsigned long long int* d_hash;
    cudaMalloc(&d_hash, sizeof(unsigned long long int));
    gpu_dct_single(d_img, d_hash);
    cudaMemcpy(d_hash_val, d_hash, sizeof(unsigned long long int), cudaMemcpyDeviceToDevice);
    cudaFree(d_hash);
}

/// <summary>
/// Performs a discrete cosine transform on the given image and stores the result in the given hash value.
/// </summary>
/// <param name="img">The image to perform the DCT on.</param>
/// <param name="d_hash_val">The hash value to store the result in.</param>
/// <returns>
/// Nothing.
/// </returns>
void GpuDct::dct(const cv::Mat& img, unsigned long long int* d_hash_val)
{
    unsigned long long int* d_hash;
    cudaMalloc(&d_hash, sizeof(unsigned long long int));
    cudaMemset(d_A, 0, m_size * m_size * sizeof(float));
    cudaMemcpy(d_A, reinterpret_cast<float*>(img.data), m_size * m_size * sizeof(float), cudaMemcpyHostToDevice);
    gpu_dct_single(d_A, d_hash);
    cudaMemcpy(d_hash_val, d_hash, sizeof(unsigned long long int), cudaMemcpyDeviceToDevice);
    cudaFree(d_hash);
}

/// <summary>
/// Calculates the DCT hash values of a vector of cv::Mat images using GPU.
/// </summary>
/// <param name="images">The images to be hashed.</param>
/// <param name="d_hash_values">The hash values of the images [gpu].</param>
/// <returns>
/// void
/// </returns>
/**
 * @brief calculate the DCT hash values of a vector of cv::Mat images
 *
 * @param images the images to be hashed
 * @param d_hash_values the hash values of the images [gpu]
 */
void GpuDct::stream_dct(std::vector<cv::Mat>& images, unsigned long long int* d_hash_values)
{
    const int b_s = 64;
    int n = images[0].size().width;
    int n_imgs = images.size();
    float* g_mat;
    cudaMalloc(&g_mat, n_imgs * n * n * sizeof(float));
    int* d_bit_arrays;
    cudaMalloc(&d_bit_arrays, n_imgs * b_s * sizeof(int));
    float* tmp;
    cudaMalloc(&tmp, n_imgs * n * n * sizeof(float));
    float* DCT;
    cudaMalloc(&DCT, n_imgs * n * n * sizeof(float));
    float* d_Transform;
    cudaMalloc(&d_Transform, n * n * sizeof(float));
    kernel_get_T<<<n, n>>>(d_Transform);
    cudaDeviceSynchronize();
    cublasHandle_t c_handle;
    cublasCreate(&c_handle);

    // allocate and initialize an array of stream handles
    std::vector<cudaStream_t> streams(n_imgs);
    for (int i = 0; i < n_imgs; i++) {
        cudaStreamCreate(&(streams[i]));
    }
    for (int i = 0; i < n_imgs; i++) {
        // do the hash matrix multiplication with streams
        cudaMemcpyAsync(g_mat + i * n * n, reinterpret_cast<float*>(images[i].data), n * n * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        gpu_dct(g_mat + i * n * n, d_bit_arrays + i * b_s, tmp + i * n * n, DCT + i * n * n, d_Transform, n, c_handle, streams[i]);
    }
    for (int i = 0; i < n_imgs; i++) {
        cudaStreamDestroy(streams[i]);
    }
    kernel_add_bits<<<n_imgs, 64>>>(d_bit_arrays, d_hash_values);

    // clean up
    cublasDestroy(c_handle);
    cudaFree(d_bit_arrays);
    cudaFree(g_mat);
    cudaFree(DCT);
    cudaFree(tmp);
    cudaFree(d_Transform);
}

/**
 * @brief calculates the DCT hash values of images - the images should be pre-loaded to gpu
 *
 * @param d_image_arr the images on gpu in contigous memory
 * @param d_hash_values the hash values on gpu
 * @param width width of image
 * @param n_imgs number of images
 */
void GpuDct::stream_dct(const float* d_image_arr, unsigned long long int* d_hash_values, int width, int n_imgs)
{
    const int b_s = 64;
    int n = width;
    int* d_bit_arrays;
    cudaMalloc(&d_bit_arrays, n_imgs * b_s * sizeof(int));
    float* tmp;
    cudaMalloc(&tmp, n_imgs * n * n * sizeof(float));
    float* DCT;
    cudaMalloc(&DCT, n_imgs * n * n * sizeof(float));
    float* d_Transform;
    cudaMalloc(&d_Transform, n * n * sizeof(float));
    kernel_get_T<<<n, n>>>(d_Transform);
    cudaDeviceSynchronize();
    cublasHandle_t c_handle;
    cublasCreate(&c_handle);

    // allocate and initialize an array of stream handles
    std::vector<cudaStream_t> streams(n_imgs);
    for (int i = 0; i < n_imgs; i++) {
        cudaStreamCreate(&(streams[i]));
    }
    for (int i = 0; i < n_imgs; i++) {
        // do the hash matrix multiplication with streams
        gpu_dct(d_image_arr + i * n * n, d_bit_arrays + i * b_s, tmp + i * n * n, DCT + i * n * n, d_Transform, n, c_handle, streams[i]);
    }
    for (int i = 0; i < n_imgs; i++) {
        cudaStreamDestroy(streams[i]);
    }
    kernel_add_bits<<<n_imgs, 64>>>(d_bit_arrays, d_hash_values);

    // clean up
    cublasDestroy(c_handle);
    cudaFree(d_bit_arrays);
    cudaFree(DCT);
    cudaFree(tmp);
    cudaFree(d_Transform);
}

unsigned long long int GpuDct::gpu_dct_single(const float* A, unsigned long long int* dhash_val)
{
    // resetting values
    const int n = m_size;

    cudaMemset(d_bits, 0, 64 * sizeof(int));
    cudaMemset(d_tmp, 0, n * n * sizeof(float));
    cudaMemset(d_DCT, 0, n * n * sizeof(float));

    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;
    // T * A * T'
    cublasSgemm(m_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, alpha, d_T, n, A, n, beta, d_tmp, n);
    cublasSgemm(m_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, d_tmp, n, d_T, n, beta, d_DCT, n);
    // binarise the 8x8 DCT values
    kernel_compute_hash<<<1, 64>>>(d_DCT, d_bits, n);
    kernel_add_bits<<<1, 64>>>(d_bits, dhash_val);
    unsigned long long int h_hash;
    cudaMemcpy(&h_hash, dhash_val, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    return h_hash;
}

// calculates the DCT of an image (needs to be fixed)
void GpuDct::gpu_dct(const float* A, int* dbit_array, float* tmp, float* DCT, float* d_Transform, const int n, cublasHandle_t handle, cudaStream_t s)
{
    const float alf = 1;
    const float bet = 0;
    const float* alpha = &alf;
    const float* beta = &bet;
    // T * A * T'
    cublasSetStream(handle, s);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, alpha, d_Transform, n, A, n, beta, tmp, n);
    cublasSetStream(handle, s);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, tmp, n, d_Transform, n, beta, DCT, n);
    // binarise the 8x8 DCT values
    kernel_compute_hash<<<1, 64, 0, s>>>(DCT, dbit_array, n);
}

void GpuDct::getDCTMatrix(const int rows, const int cols, float* d_Transfrom)
{
    kernel_get_T<<<rows, cols>>>(d_Transfrom);
}