#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <gpu_dct.cuh>

using namespace gpu_dct;

namespace {

class CudaEvent {
  public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault)
        : m_event(nullptr) {
        throw_on_cuda_error(cudaEventCreateWithFlags(&m_event, flags),
                            "Failed to create CUDA event");
    }

    ~CudaEvent() {
        if (m_event != nullptr) {
            cudaEventDestroy(m_event);
        }
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    CudaEvent(CudaEvent &&other) noexcept
        : m_event(std::exchange(other.m_event, nullptr)) {}

    CudaEvent &operator=(CudaEvent &&other) noexcept {
        if (this != &other) {
            if (m_event != nullptr) {
                cudaEventDestroy(m_event);
            }
            m_event = std::exchange(other.m_event, nullptr);
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t get() const { return m_event; }

  private:
    cudaEvent_t m_event;
};

class GpuTimer {
  public:
    explicit GpuTimer(cudaStream_t stream) : m_stream(stream) {}

    void start() {
        throw_on_cuda_error(cudaEventRecord(m_start.get(), m_stream),
                            "Failed to record start event");
    }

    [[nodiscard]] double stop_ms() {
        throw_on_cuda_error(cudaEventRecord(m_stop.get(), m_stream),
                            "Failed to record stop event");
        throw_on_cuda_error(cudaEventSynchronize(m_stop.get()),
                            "Failed to synchronize stop event");
        float elapsed_ms = 0.0f;
        throw_on_cuda_error(
            cudaEventElapsedTime(&elapsed_ms, m_start.get(), m_stop.get()),
            "Failed to compute elapsed time");
        return static_cast<double>(elapsed_ms);
    }

  private:
    cudaStream_t m_stream;
    CudaEvent m_start{};
    CudaEvent m_stop{};
};

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t element_count) { allocate(element_count); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : m_ptr(std::exchange(other.m_ptr, nullptr)) {}

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            m_ptr = std::exchange(other.m_ptr, nullptr);
        }
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    void allocate(size_t element_count) {
        reset();
        if (element_count == 0) {
            return;
        }
        throw_on_cuda_error(cudaMalloc(reinterpret_cast<void **>(&m_ptr),
                                       element_count * sizeof(T)),
                            "Failed to allocate device buffer");
    }

    void reset() {
        if (m_ptr != nullptr) {
            cudaFree(m_ptr);
            m_ptr = nullptr;
        }
    }

    [[nodiscard]] T *get() { return m_ptr; }
    [[nodiscard]] const T *get() const { return m_ptr; }

  private:
    T *m_ptr{nullptr};
};

class StreamList {
  public:
    explicit StreamList(int count) : m_streams(count, nullptr) {
        for (int i = 0; i < count; ++i) {
            try {
                throw_on_cuda_error(cudaStreamCreate(&m_streams[i]),
                                    "Failed to create CUDA stream");
            } catch (...) {
                for (int j = 0; j < i; ++j) {
                    if (m_streams[j] != nullptr) {
                        cudaStreamDestroy(m_streams[j]);
                        m_streams[j] = nullptr;
                    }
                }
                throw;
            }
        }
    }

    ~StreamList() { release(); }

    StreamList(const StreamList &) = delete;
    StreamList &operator=(const StreamList &) = delete;

    StreamList(StreamList &&other) noexcept
        : m_streams(std::move(other.m_streams)) {
        other.m_streams.clear();
    }

    StreamList &operator=(StreamList &&other) noexcept {
        if (this != &other) {
            release();
            m_streams = std::move(other.m_streams);
            other.m_streams.clear();
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t *data() { return m_streams.data(); }
    [[nodiscard]] const cudaStream_t *data() const { return m_streams.data(); }
    [[nodiscard]] int size() const {
        return static_cast<int>(m_streams.size());
    }
    [[nodiscard]] cudaStream_t operator[](int index) const {
        return m_streams[index];
    }
    [[nodiscard]] cudaStream_t &operator[](int index) {
        return m_streams[index];
    }

  private:
    void release() {
        for (auto &stream : m_streams) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
    }

    std::vector<cudaStream_t> m_streams;
};

} // namespace

void print_header(const std::string &title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_stats(const std::string &test_name, int batch_size, double time_ms) {
    double time_per_image_us = (time_ms * 1000.0) / batch_size;
    double throughput = batch_size / (time_ms / 1000.0);

    std::cout << std::left << std::setw(35) << test_name << std::right
              << std::setw(10) << std::fixed << std::setprecision(3) << time_ms
              << " ms  " << std::setw(10) << std::fixed << std::setprecision(2)
              << time_per_image_us << " μs/img  " << std::setw(12) << std::fixed
              << std::setprecision(0) << throughput << " img/s" << std::endl;
}

template <typename T>
void benchmark_single_image(int n, int iterations = 1000) {
    print_header("Single Image Performance - " + std::string(typeid(T).name()));

    GpuDct<T> dct(n);
    std::vector<T> h_image(n * n);

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (auto &val : h_image) {
        val = static_cast<T>(dis(gen));
    }

    // Warmup
    uint64_t warmup_hash = 0;
    for (int i = 0; i < 10; ++i) {
        warmup_hash = dct.dct_host(h_image.data());
    }
    throw_on_cuda_error(cudaDeviceSynchronize(),
                        "Single-image warmup synchronize failed");

    // Benchmark (GPU time via stream events)
    GpuTimer timer(dct.stream());
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        warmup_hash = dct.dct_host(h_image.data());
    }
    double time_ms = timer.stop_ms();
    static_cast<void>(warmup_hash);

    print_stats("Single image (default stream)", iterations, time_ms);

    std::cout << "\nMatrix size: " << n << "x" << n << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
}

template <typename T>
void benchmark_batch(int n, int batch_size, int iterations = 100) {
    print_header("Batch Processing Performance (" + std::to_string(batch_size) +
                 " images)");

    GpuDct<T> dct(n);
    std::vector<T> h_images(n * n * batch_size);
    std::vector<uint64_t> h_hashes(batch_size);

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (auto &val : h_images) {
        val = static_cast<T>(dis(gen));
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
    }
    throw_on_cuda_error(cudaDeviceSynchronize(),
                        "Batch warmup synchronize failed");

    // Benchmark
    GpuTimer timer(dct.stream());
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
    }
    double time_ms = timer.stop_ms();

    print_stats("Batch processing (fused kernel)", batch_size * iterations,
                time_ms);

    std::cout << "\nMatrix size: " << n << "x" << n << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Total batches: " << iterations << std::endl;
    std::cout << "Total images: " << (batch_size * iterations) << std::endl;
}

template <typename T>
void benchmark_multistream(int n, int batch_size, int num_streams) {
    print_header("Multi-Stream Performance (" + std::to_string(batch_size) +
                 " images, " + std::to_string(num_streams) + " streams)");

    if (num_streams <= 0) {
        throw std::invalid_argument(
            "benchmark_multistream: num_streams must be positive");
    }

    GpuDct<T> dct(n);

    DeviceBuffer<T> d_images(static_cast<size_t>(n) * n * batch_size);
    DeviceBuffer<uint64_t> d_hashes(batch_size);

    std::vector<T> h_images(static_cast<size_t>(n) * n * batch_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 255.0);
    for (auto &val : h_images) {
        val = static_cast<T>(dis(gen));
    }

    throw_on_cuda_error(cudaMemcpy(d_images.get(), h_images.data(),
                                   h_images.size() * sizeof(T),
                                   cudaMemcpyHostToDevice),
                        "Failed to upload images to device");

    StreamList streams(num_streams);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        dct.batch_dct_device_multistream(d_images.get(), d_hashes.get(),
                                         batch_size, streams.data(),
                                         streams.size());
    }
    throw_on_cuda_error(cudaDeviceSynchronize(),
                        "Multi-stream warmup synchronize failed");

    const int iterations = 100;

    std::vector<CudaEvent> start_events;
    start_events.reserve(static_cast<size_t>(streams.size()));
    for (int i = 0; i < streams.size(); ++i) {
        start_events.emplace_back();
        throw_on_cuda_error(
            cudaEventRecord(start_events.back().get(), streams[i]),
            "Failed to record stream start event");
    }

    for (int i = 0; i < iterations; ++i) {
        dct.batch_dct_device_multistream(d_images.get(), d_hashes.get(),
                                         batch_size, streams.data(),
                                         streams.size());
    }

    std::vector<CudaEvent> stop_events;
    stop_events.reserve(static_cast<size_t>(streams.size()));
    for (int i = 0; i < streams.size(); ++i) {
        stop_events.emplace_back();
        throw_on_cuda_error(
            cudaEventRecord(stop_events.back().get(), streams[i]),
            "Failed to record stream stop event");
    }

    double max_stream_ms = 0.0;
    for (int i = 0; i < streams.size(); ++i) {
        throw_on_cuda_error(cudaEventSynchronize(stop_events[i].get()),
                            "Failed to synchronize stream stop event");
        float elapsed_ms = 0.0f;
        throw_on_cuda_error(cudaEventElapsedTime(&elapsed_ms,
                                                 start_events[i].get(),
                                                 stop_events[i].get()),
                            "Failed to compute stream elapsed time");
        max_stream_ms =
            std::max(max_stream_ms, static_cast<double>(elapsed_ms));
    }

    print_stats("Multi-stream (fused kernel)", batch_size * iterations,
                max_stream_ms);

    std::cout << "\nMatrix size: " << n << "x" << n << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of streams: " << num_streams << std::endl;
    std::cout << "Total iterations: " << iterations << std::endl;
    std::cout << "Total images: " << (batch_size * iterations) << std::endl;
}

void benchmark_comparison() {
    print_header("Performance Comparison: Different Matrix Sizes");

    std::cout << "\n"
              << std::left << std::setw(20) << "Matrix Size" << std::right
              << std::setw(15) << "Time (ms)" << std::setw(18)
              << "Time/Image (μs)" << std::setw(18) << "Throughput (K/s)"
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
        throw_on_cuda_error(cudaDeviceSynchronize(),
                            "Comparison warmup synchronize failed");

        // Benchmark
        GpuTimer timer(dct.stream());
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), batch_size);
        double time_ms = timer.stop_ms();

        double time_per_image_us = (time_ms * 1000.0) / batch_size;
        double throughput_k = batch_size / time_ms;

        std::cout << std::left << std::setw(20)
                  << (std::to_string(n) + "x" + std::to_string(n)) << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << time_ms << std::setw(18) << std::fixed
                  << std::setprecision(2) << time_per_image_us << std::setw(18)
                  << std::fixed << std::setprecision(1) << throughput_k
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
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor
              << std::endl;
    std::cout << "Total Global Memory: "
              << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Shared Memory Per Block: " << (prop.sharedMemPerBlock / 1024)
              << " KB" << std::endl;
    std::cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock
              << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    int memory_clock_khz = 0;
    if (cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate,
                               device) != cudaSuccess) {
        memory_clock_khz = 0;
    }
    if (memory_clock_khz > 0) {
        std::cout << "Memory Clock Rate: " << (memory_clock_khz / 1000)
                  << " MHz" << std::endl;
    } else {
        std::cout << "Memory Clock Rate: N/A" << std::endl;
    }

    int memory_bus_width = 0;
    if (cudaDeviceGetAttribute(&memory_bus_width,
                               cudaDevAttrGlobalMemoryBusWidth,
                               device) != cudaSuccess) {
        memory_bus_width = 0;
    }
    if (memory_bus_width > 0) {
        std::cout << "Memory Bus Width: " << memory_bus_width << " bits"
                  << std::endl;
    } else {
        std::cout << "Memory Bus Width: N/A" << std::endl;
    }
}

int main(int argc, char **argv) {
    int base_size = 32;
    if (argc > 1) {
        char *end_ptr = nullptr;
        long parsed = std::strtol(argv[1], &end_ptr, 10);
        if (end_ptr == argv[1] || *end_ptr != '\0') {
            std::cerr << "Invalid size argument: " << argv[1] << std::endl;
            return 1;
        }
        base_size = static_cast<int>(parsed);
    }

    constexpr std::array<int, 4> supported_sizes{32, 64, 128, 256};
    if (std::ranges::find(supported_sizes, base_size) ==
        supported_sizes.end()) {
        std::cerr << "Unsupported size: " << base_size
                  << ". Supported sizes are 32, 64, 128, 256." << std::endl;
        return 1;
    }

    std::cout << "\n";
    std::cout
        << "╔══════════════════════════════════════════════════════════════"
           "═════╗\n";
    std::cout
        << "║         GpuDct Performance Benchmark - Fused Kernel Edition  "
           "     ║\n";
    std::cout
        << "╚══════════════════════════════════════════════════════════════"
           "═════╝\n";

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
    std::cout << "\n"
              << std::left << std::setw(20) << "Data Type" << std::right
              << std::setw(15) << "Time (ms)" << std::setw(18)
              << "Time/Image (μs)" << std::endl;
    std::cout << std::string(53, '-') << std::endl;

    {
        GpuDct<float> dct(base_size);
        std::vector<float> h_images(base_size * base_size * 1000);
        std::vector<uint64_t> h_hashes(1000);
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        throw_on_cuda_error(cudaDeviceSynchronize(),
                            "Float warmup synchronize failed");

        GpuTimer timer(dct.stream());
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        double time_ms = timer.stop_ms();

        std::cout << std::left << std::setw(20) << "float" << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << time_ms << std::setw(18) << std::fixed
                  << std::setprecision(2) << (time_ms * 1000.0 / 1000)
                  << std::endl;
    }

    {
        GpuDct<double> dct(base_size);
        std::vector<double> h_images(base_size * base_size * 1000);
        std::vector<uint64_t> h_hashes(1000);
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        throw_on_cuda_error(cudaDeviceSynchronize(),
                            "Double warmup synchronize failed");

        GpuTimer timer(dct.stream());
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        double time_ms = timer.stop_ms();

        std::cout << std::left << std::setw(20) << "double" << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << time_ms << std::setw(18) << std::fixed
                  << std::setprecision(2) << (time_ms * 1000.0 / 1000)
                  << std::endl;
    }

    {
        GpuDct<int> dct(base_size);
        std::vector<int> h_images(base_size * base_size * 1000);
        std::vector<uint64_t> h_hashes(1000);
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        throw_on_cuda_error(cudaDeviceSynchronize(),
                            "Int warmup synchronize failed");

        GpuTimer timer(dct.stream());
        timer.start();
        dct.batch_dct_host(h_images.data(), h_hashes.data(), 1000);
        double time_ms = timer.stop_ms();

        std::cout << std::left << std::setw(20) << "int (->float)" << std::right
                  << std::setw(15) << std::fixed << std::setprecision(3)
                  << time_ms << std::setw(18) << std::fixed
                  << std::setprecision(2) << (time_ms * 1000.0 / 1000)
                  << std::endl;
    }

    std::cout << "\n";
    std::cout
        << "╔══════════════════════════════════════════════════════════════"
           "═════╗\n";
    std::cout
        << "║                    Benchmark Complete!                       "
           "     ║\n";
    std::cout
        << "╚══════════════════════════════════════════════════════════════"
           "═════╝\n";
    std::cout << std::endl;

    return 0;
}
