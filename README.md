# GpuDct: CUDA DCT Hashing Library

GpuDct is a CUDA C++20 library that computes 64-bit perceptual hashes from square images using fused Discrete Cosine Transform (DCT) kernels. Each kernel evaluates the full T * A * T' pipeline, extracts the 8x8 low-frequency block on device, and emits a median-threshold signature without extra launches or host round trips.

## Highlights
- Fused single-pass kernels for 32, 64, 128, and 256 sized images with constant-memory transforms
- Stream-ordered temporary allocations via CUDA memory pools (no hot-path malloc)
- In-kernel 8x8 hashing and median selection yielding a 64-bit binary fingerprint
- Batch and multi-stream helpers for high-throughput pipelines
- Benchmarks instrumented with CUDA events for precise GPU time attribution
- CMake package configured for CUDA + C++20, friendly with FetchContent and install exports

## Requirements
- NVIDIA GPU with compute capability 7.5 or newer (tune `CMAKE_CUDA_ARCHITECTURES` as needed)
- CUDA Toolkit 12.x (tested) with `nvcc`
- CMake 3.18 or newer
- Host compiler with full C++20 support (GCC 11+, Clang 14+, MSVC 19.3+)
- No bundled image-processing dependencies. Provide your own contiguous buffers from any loader you prefer (stb_image, OpenCV, etc.).

## Quick Start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Outputs include `libGpuDct.a` and sample binaries under `build/examples/`. Sanity check performance with:

```bash
./build/examples/gpu_dct_benchmark        # defaults to 32x32
./build/examples/gpu_dct_benchmark 256    # alternate size
```

## Basic Tutorial

The primary entry point is `gpu_dct::GpuDct<T>`. Supported image sizes are 32, 64, 128, and 256.

### 1. Single image hashing from host memory

```cpp
#include <gpu_dct.cuh>
#include <vector>
#include <cstdint>
#include <iostream>

int main() {
    constexpr int N = 32;
    gpu_dct::GpuDct<float> dct(N);

    std::vector<float> image(N * N);
    for (size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<float>(i % 256);
    }

    const uint64_t hash = dct.dct_host(image.data());
    std::cout << "hash: 0x" << std::hex << hash << std::dec << "\n";
    return 0;
}
```

`dct_host` is synchronous and optionally accepts a CUDA stream to integrate with existing GPU work.

### 2. Batched host processing

```cpp
constexpr int N = 64;
constexpr int batch = 16;
gpu_dct::GpuDct<float> dct(N);

std::vector<float> images(static_cast<size_t>(N) * N * batch);
std::vector<uint64_t> hashes(batch);

dct.batch_dct_host(images.data(), hashes.data(), batch);
```

The helper stages data through stream-ordered pools, launches fused kernels for the entire batch, and returns once hashes are copied back.

### 3. Device-to-device workflows and multi-stream execution

```cpp
#include <array>

gpu_dct::GpuDct<float> dct(128);
constexpr int batch = 64;

float* d_images = nullptr;
uint64_t* d_hashes = nullptr;
cudaMalloc(&d_images, 128 * 128 * batch * sizeof(float));
cudaMalloc(&d_hashes, batch * sizeof(uint64_t));

// populate d_images on device...

dct.batch_dct_device(d_images, d_hashes, batch);

std::array<cudaStream_t, 4> streams{};
for (auto& s : streams) {
    cudaStreamCreate(&s);
}

dct.batch_dct_device_multistream(d_images, d_hashes, batch, streams);

for (auto s : streams) {
    cudaStreamDestroy(s);
}

cudaFree(d_images);
cudaFree(d_hashes);
```

Hashes remain on the device, enabling additional GPU-side comparisons before any host transfer.

### 4. Hashing a real image

Download any public grayscale or RGB square image and feed it through the helper utility:

```bash
cmake --build build -j$(nproc)
./build/examples/gpu_dct_hash_image path/to/lena.jpg 256
```

The tool uses stb_image to decode the asset, converts it to grayscale, downsamples to the requested DCT size (32, 64, 128, or 256), and prints the 64-bit perceptual hash so you can cross-check against other implementations.

### Feeding data from image libraries (optional)

GpuDct only expects a contiguous buffer of pixel intensities, so you can lift data from whatever host-side library you already use without additional dependencies. For example, with OpenCV:

```cpp
cv::Mat gray = cv::imread(path, cv::IMREAD_GRAYSCALE);
if (!gray.data || gray.rows != N || gray.cols != N) {
    throw std::runtime_error("unexpected image dimensions");
}

std::vector<float> image(gray.rows * gray.cols);
std::transform(gray.begin<uint8_t>(), gray.end<uint8_t>(), image.begin(),
               [](uint8_t v) { return static_cast<float>(v); });

gpu_dct::GpuDct<float> dct(N);
const uint64_t hash = dct.dct_host(image.data());
```

Any loader that produces a contiguous block (stb_image, libpng, custom CUDA pipelines) can be wired up the same way.

## Using GpuDct in another CMake project

```cmake
include(FetchContent)
FetchContent_Declare(
    GpuDct
    GIT_REPOSITORY https://github.com/tenxlenx/GpuDct.git
    GIT_TAG main
)

FetchContent_MakeAvailable(GpuDct)

add_executable(hash_demo main.cpp)
target_link_libraries(hash_demo PRIVATE GpuDct CUDA::cudart)
set_property(TARGET hash_demo PROPERTY CXX_STANDARD 20)
```

Override `CMAKE_CUDA_ARCHITECTURES` in the parent project to match deployment hardware.

## Benchmarking

`examples/gpu_dct_benchmark` exercises single images, batched runs, and multi-stream scenarios with CUDA event profiling on every test. CLI usage:

```
./gpu_dct_benchmark                # 32x32, default iterations
./gpu_dct_benchmark 128            # choose image size
./gpu_dct_benchmark 64 --streams 4 # adjust streams or iterations
```

The tool reports per-image latency, throughput, and data type comparisons for quick regression checks.

## Troubleshooting
- Mismatch between compiled and runtime GPU architectures: set `CMAKE_CUDA_ARCHITECTURES` explicitly.
- Out-of-memory during large batches: raise the CUDA malloc heap limit or reduce concurrent streams.
- Integrating with pre-existing CUDA streams: pass your stream to constructors or method overloads to preserve ordering.

## License

MIT. See `LICENSE` for details.

The repository vendors `stb_image.h` (public-domain / MIT dual licensed) in `third_party/` for sample image decoding.
