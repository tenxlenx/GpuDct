cmake_minimum_required(VERSION 3.26)
project(GpuDct CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} STATIC src/gpu_dct.cu
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
target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Set output directory for the library file
set_target_properties(${PROJECT_NAME} PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib
)
install(DIRECTORY include/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include)
