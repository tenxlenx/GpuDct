#include <gpu_dct.cuh>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <ranges>
#include <string>

using namespace gpu_dct;

class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    
    double stop_us() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end_time - start_time).count();
    }
};

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_stats(const std::string& test_name, int batch_size, double time_ms) {
    double time_per_image_us = (time_ms * 1000.0) / batch_size;
    double throughput = batch_size / (time_ms / 1000.0);
    
    std::cout << std::left << std::setw(35) << test_name
              << std::right 
              << std::setw(10) << std::fixed << std::setprecision(3) << time_ms << " ms  "
              << std::setw(10) << std::fixed << std::setprecision(2) << time_per_image_us << " μs/img  "
              << std::setw(12) << std::fixed << std::setprecision(0) << throughput << " img/s"
              << std::endl;
}

template<typename T>
void benchmark_single_image(int n, int iterations = 1000) {
    print_header("Single Image Performance - " + std::string(typeid(T).name()));
    
    GpuDct<T> dct(n);
    std::vector<T> h_image(n * n);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (auto& val : h_image) {
        val = static_cast<T>(dis(gen));
    }
    
    // Warmup
    uint64_t warmup_hash = 0;
    for (int i = 0; i < 10; i++) {
        warmup_hash = dct.dct_host(h_image.data());
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        warmup_hash = dct.dct_host(h_image.data());
    }
    cudaDeviceSynchronize();
    double time_ms = timer.stop_ms();
    
    print_stats("Single image (default stream)", iterations, time_ms);
    
    std::cout << "\nMatrix size: " << n << "x" << n << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
}

template<typename T>
void benchmark_batch(int n, int batch_size, int iterations = 100) {
    print_header("Batch Processing Performance (" + std::to_string(batch_size) + " images)");
    
    GpuDct<T> dct(n);
    std::vector<T> h_images(n * n * batch_size);
    std::vector<uint64_t> h_hashes(batch_size);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (auto& val : h_images) {
        val = static_cast<T>(dis(gen));
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
    }
    cudaDeviceSynchronize();
    double time_ms = timer.stop_ms();
    
    print_stats("Batch processing (fused kernel)", batch_size * iterations, time_ms);
    
    std::cout << "\nMatrix size: " << n << "x" << n << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Total batches: " << iterations << std::endl;
    std::cout << "Total images: " << (batch_size * iterations) << std::endl;
}

template<typename T>
void benchmark_multistream(int n, int batch_size, int num_streams) {
    print_header("Multi-Stream Performance (" + std::to_string(batch_size) + " images, " + 
                 std::to_string(num_streams) + " streams)");
    
    GpuDct<T> dct(n);
    
    // Allocate device memory
    T* d_images;
    uint64_t* d_hashes;
    cudaMalloc(&d_images, n * n * batch_size * sizeof(T));
    cudaMalloc(&d_hashes, batch_size * sizeof(uint64_t));
    
    // Fill with random data
    std::vector<T> h_images(n * n * batch_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (auto& val : h_images) {
        val = static_cast<T>(dis(gen));
    }
    cudaMemcpy(d_images, h_images.data(), n * n * batch_size * sizeof(T), cudaMemcpyHostToDevice);
    
    // Create streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Warmup
    for (int i = 0; i < 5; i++) {
        dct.batch_dct_device_multistream(d_images, d_hashes, batch_size, streams.data(), num_streams);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    int iterations = 100;
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        dct.batch_dct_device_multistream(d_images, d_hashes, batch_size, streams.data(), num_streams);
    }
    cudaDeviceSynchronize();
    double time_ms = timer.stop_ms();
    
    print_stats("Multi-stream (fused kernel)", batch_size * iterations, time_ms);
    
    std::cout << "\nMatrix size: " << n << "x" << n << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of streams: " << num_streams << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
    std::cout << "Total images: " << (batch_size * iterations) << std::endl;
    
    // Cleanup
    for (auto& s : streams) {
        cudaStreamDestroy(s);
    }
    cudaFree(d_images);
    cudaFree(d_hashes);
}

void benchmark_comparison() {
    print_header("Performance Comparison: Different Matrix Sizes");
    
    std::cout << "\n" << std::left << std::setw(20) << "Matrix Size"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Time/Image (μs)"
              << std::setw(18) << "Throughput (K/s)"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<int> sizes = {32, 64, 128, 256};
    int batch_size = 256;
    
    for (int n : sizes) {
        GpuDct<float> dct(n);
        std::vector<float> h_images(n * n * batch_size);
    std::vector<uint64_t> h_hashes(batch_size);
        
        // Warmup
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
        cudaDeviceSynchronize();
        
        // Benchmark
        Timer timer;
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
        cudaDeviceSynchronize();
        double time_ms = timer.stop_ms();
        
        double time_per_image_us = (time_ms * 1000.0) / batch_size;
        double throughput_k = batch_size / time_ms;
        
        std::cout << std::left << std::setw(20) << (std::to_string(n) + "x" + std::to_string(n))
                  << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << time_per_image_us
                  << std::setw(18) << std::fixed << std::setprecision(1) << throughput_k
                  << std::endl;
    }
}

void print_device_info() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    print_header("GPU Device Information");
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Shared Memory Per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    int memory_clock_khz = 0;
    if (cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess) {
        memory_clock_khz = 0;
    }
    if (memory_clock_khz > 0) {
        std::cout << "Memory Clock Rate: " << (memory_clock_khz / 1000) << " MHz" << std::endl;
    } else {
        std::cout << "Memory Clock Rate: N/A" << std::endl;
    }

    int memory_bus_width = 0;
    if (cudaDeviceGetAttribute(&memory_bus_width, cudaDevAttrGlobalMemoryBusWidth, device) != cudaSuccess) {
        memory_bus_width = 0;
    }
    if (memory_bus_width > 0) {
        std::cout << "Memory Bus Width: " << memory_bus_width << " bits" << std::endl;
    } else {
        std::cout << "Memory Bus Width: N/A" << std::endl;
    }
}

int main(int argc, char** argv) {
    int base_size = 32;
    if (argc > 1) {
        char* end_ptr = nullptr;
        long parsed = std::strtol(argv[1], &end_ptr, 10);
        if (end_ptr == argv[1] || *end_ptr != '\0') {
            std::cerr << "Invalid size argument: " << argv[1] << std::endl;
            return 1;
        }
        base_size = static_cast<int>(parsed);
    }

    constexpr std::array<int, 4> supported_sizes{32, 64, 128, 256};
    if (std::ranges::find(supported_sizes, base_size) == supported_sizes.end()) {
        std::cerr << "Unsupported size: " << base_size
                  << ". Supported sizes are 32, 64, 128, 256." << std::endl;
        return 1;
    }

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         GpuDct Performance Benchmark - Fused Kernel Edition       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    
    print_device_info();
    
    // Single image benchmarks
    benchmark_single_image<float>(base_size, 1000);
    
    // Batch benchmarks
    benchmark_batch<float>(base_size, 100, 100);
    benchmark_batch<float>(base_size, 1000, 50);
    benchmark_batch<float>(base_size, 10000, 10);
    
    // Multi-stream benchmarks
    benchmark_multistream<float>(base_size, 1000, 2);
    benchmark_multistream<float>(base_size, 1000, 4);
    benchmark_multistream<float>(base_size, 1000, 8);
    
    // Size comparison
    benchmark_comparison();
    
    // Different data types
    print_header("Data Type Comparison (" + std::to_string(base_size) + "x" +
                 std::to_string(base_size) + ", 1000 images)");
    std::cout << "\n" << std::left << std::setw(20) << "Data Type"
              << std::right << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Time/Image (μs)"
              << std::endl;
    std::cout << std::string(53, '-') << std::endl;
    
    {
    GpuDct<float> dct(base_size);
    std::vector<float> h_images(base_size * base_size * 1000);
        std::vector<uint64_t> h_hashes(1000);
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        cudaDeviceSynchronize();
        
        Timer timer;
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        cudaDeviceSynchronize();
        double time_ms = timer.stop_ms();
        
        std::cout << std::left << std::setw(20) << "float"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << (time_ms * 1000.0 / 1000)
                  << std::endl;
    }
    
    {
    GpuDct<double> dct(base_size);
    std::vector<double> h_images(base_size * base_size * 1000);
        std::vector<uint64_t> h_hashes(1000);
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        cudaDeviceSynchronize();
        
        Timer timer;
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        cudaDeviceSynchronize();
        double time_ms = timer.stop_ms();
        
        std::cout << std::left << std::setw(20) << "double"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << (time_ms * 1000.0 / 1000)
                  << std::endl;
    }
    
    {
    GpuDct<int> dct(base_size);
    std::vector<int> h_images(base_size * base_size * 1000);
        std::vector<uint64_t> h_hashes(1000);
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        cudaDeviceSynchronize();
        
        Timer timer;
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        cudaDeviceSynchronize();
        double time_ms = timer.stop_ms();
        
        std::cout << std::left << std::setw(20) << "int (->float)"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(3) << time_ms
                  << std::setw(18) << std::fixed << std::setprecision(2) << (time_ms * 1000.0 / 1000)
                  << std::endl;
    }
    
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    Benchmark Complete!                            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nKey Optimizations:\n";
    std::cout << "  ✓ Fused T*A*T' kernel (eliminates intermediate storage)\n";
    std::cout << "  ✓ Stream-ordered memory allocation (zero malloc overhead)\n";
    std::cout << "  ✓ Constant memory for DCT matrix (fast cached access)\n";
    std::cout << "  ✓ Warp-level optimization for 32x32 matrices\n";
    std::cout << "  ✓ Multi-stream parallelism support\n";
    std::cout << std::endl;
    
    return 0;
}
