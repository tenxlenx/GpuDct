# GpuDct: CUDA-accelerated DCT Hash Calculation Library

GpuDct is a C++ library that enables the fast extraction of Discrete Cosine Transform (DCT) hashes using NVIDIA's CUDA architecture. This library leverages GPU acceleration to provide efficient and high-performance hash computation.

## Features
- GPU-accelerated DCT hash computation
- Built with C++ and CUDA
- Utilizes cuBLAS and other CUDA libraries for optimization
- Seamless integration with OpenCV
- Easy-to-use CMake configuration

## Requirements
- CUDA Toolkit
- OpenCV
- CMake (version 3.26 or you can set it lower)
- A compatible NVIDIA GPU (Architecture 8.9 is set but you can edit it)

## Installation

To incorporate GpuDct into your project, you can use CMake's FetchContent module. You can fork this project and setup with your project URL. The example below demonstrates how to set up a simple CMake project that depends on GpuDct.

```cmake
cmake_minimum_required(VERSION 3.26)
project(testproject CUDA CXX)

# Set CUDA architecture and standard
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)

# Find required packages
find_package(OpenCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories for dependencies
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Fetch GpuDct from its Git repository
include(FetchContent)
FetchContent_Declare(
        GpuDct
        GIT_REPOSITORY https://github.com/tenxlenx/GpuDct.git
        GIT_TAG main
)

# Configure and build GpuDct if it is not already populated
FetchContent_GetProperties(GpuDct)
if(NOT GpuDct_POPULATED)
    FetchContent_Populate(GpuDct)
    add_subdirectory(${gpudct_SOURCE_DIR} ${gpudct_BINARY_DIR})
endif()

# Create the main project executable
add_executable(${PROJECT_NAME} main.cpp)

# Enable CUDA separable compilation
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        GpuDct
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cufft
        CUDA::cusolver
        CUDA::cuda_driver
)

# Specify target include directories
target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${gpudct_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
