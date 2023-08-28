cmake_minimum_required(VERSION 3.26)
project(GpuDctLib CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(GpuDctLib STATIC src/gpu_dct.cu
        include/gpu_dct.cuh)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cufft
        CUDA::cusolver
        CUDA::cuda_driver)


# Set include directories for your library
target_include_directories(GpuDctLib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Set output directory for the library file
set_target_properties(GpuDctLib PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
)

