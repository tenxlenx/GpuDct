# Example executable
add_executable(gpu_dct_example example_usage.cu)
target_link_libraries(gpu_dct_example PRIVATE GpuDct)
set_target_properties(gpu_dct_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Benchmark executable  
add_executable(gpu_dct_benchmark benchmark.cu)
target_link_libraries(gpu_dct_benchmark PRIVATE GpuDct)
set_target_properties(gpu_dct_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
