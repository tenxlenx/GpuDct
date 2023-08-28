#pragma once
#include "cublas_v2.h"
#include <vector>
#include <opencv2/core.hpp>


class GpuDct {

    public:
    explicit GpuDct(int n);

    ~GpuDct();

    void setHandle(const cublasHandle_t &_handle);
    

    /**
     * @brief calculate the dct hash value of a single image and return the hash value 
     * 
     * @param img image
     * @return hash value
     */
    unsigned long long int dct(const cv::Mat &img);

    /**
     * @brief calculates dct hash 
     * 
     * @param d_img GPU image
     * @param d_hash GPU hash value hash gets copied to user supplied hash value - need to allocate memory on calling side
     * @param img_width width of image
     */
    void dct(const float *d_img, unsigned long long int *d_hash_val);

    /**
     * @brief 
     * 
     * @param img image to upload and hash
     * @param d_hash_val device hash value, need to be allocated on the caller side
     */
    void dct(const cv::Mat &img, unsigned long long int *d_hash_val);

    /**
     * @brief calculate the DCT hash values of a vector of cv::Mat images
     * 
     * @param images the images to be hashed
     * @param d_hash_values the hash values of the images [gpu]
     */
    static void stream_dct(std::vector<cv::Mat> &images, unsigned long long int *d_hash_values);

    /**
     * @brief calculates the DCT hash values of images - the images should be pre-loaded to gpu
     * 
     * @param d_image_arr the images on gpu in contigous memory
     * @param d_hash_values the hash values on gpu
     * @param width width of image
     * @param n_imgs number of images
     */
    static void stream_dct(const float *d_image_arr, unsigned long long int *d_hash_values, int width, int n_imgs);


    private:
    cublasHandle_t m_handle;
    int m_size;
    float *d_T;
    float *d_A;   
    int *d_bits; 
    float *d_tmp; 
    float *d_DCT; 

    unsigned long long int gpu_dct_single(const float *A, unsigned long long int *dhash_val);

     // calculates the DCT of an image (needs to be fixed)
    static void gpu_dct(const float *A, int *dbit_array, float *tmp, float *DCT, float *d_Transform, int n, cublasHandle_t handle,  cudaStream_t s = nullptr);

    static void getDCTMatrix(int rows,int cols, float *d_Transfrom);
};
