# GpuDct
A library to extract DCT hashes with CUDA

A test project can build like this as an example where testproject is
the project's name and also the executable was 
named to ${PROJECT_NAME} for convenience:

```cmake

cmake_minimum_required(VERSION 3.26)
project(testproject CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)

#### dependencies = CUDA, OPENCV ####
find_package(OpenCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

#### get GpuDct #######
include(FetchContent)
FetchContent_Declare(
        GpuDct
        GIT_REPOSITORY https://github.com/tenxlenx/GpuDct.git
        GIT_TAG main
)
# Configure and build GpuDct
FetchContent_GetProperties(GpuDct)
if(NOT GpuDct_POPULATED)
    FetchContent_Populate(GpuDct)
    add_subdirectory(${gpudct_SOURCE_DIR} ${gpudct_BINARY_DIR})
endif()

#  main project
add_executable(${PROJECT_NAME} main.cpp)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE
        GpuDct
        ${OpenCV_LIBS}
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        CUDA::cufft
        CUDA::cusolver
        CUDA::cuda_driver)

# Set up target include directories for GpuDct headers
target_include_directories(${PROJECT_NAME} PRIVATE
        $<BUILD_INTERFACE:${gpudct_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
```