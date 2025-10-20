#include <chrono>
#include <cstdint>
#include <gpu_dct.cuh>
#include <iostream>
#include <vector>

using namespace gpu_dct;

// Example 1: Single image DCT with float
void example_single_float() {
  std::cout << "=== Example 1: Single Float Image ===" << std::endl;

  const int N = 32;
  GpuDct<float> dct(N);

  // Create test image
  std::vector<float> h_image(N * N);
  for (int i = 0; i < N * N; i++) {
    h_image[i] = static_cast<float>(i % 256);
  }

  // Compute hash (stream-ordered memory allocated automatically)
  uint64_t hash = dct.dct_host(h_image.data());

  std::cout << "Hash: 0x" << std::hex << hash << std::dec << std::endl;
}

// Example 2: Single image DCT with double precision
void example_single_double() {
  std::cout << "\n=== Example 2: Single Double Image ===" << std::endl;

  const int N = 32;
  GpuDct<double> dct(N);

  // Create test image
  std::vector<double> h_image(N * N);
  for (int i = 0; i < N * N; i++) {
    h_image[i] = static_cast<double>(i % 256);
  }

  uint64_t hash = dct.dct_host(h_image.data());

  std::cout << "Hash: 0x" << std::hex << hash << std::dec << std::endl;
}

// Example 3: Integer image (converted to float automatically)
void example_integer() {
  std::cout << "\n=== Example 3: Integer Image (auto-converted to float) ==="
            << std::endl;

  const int N = 32;
  GpuDct<int> dct(N);

  // Create test image with integers
  std::vector<int> h_image(N * N);
  for (int i = 0; i < N * N; i++) {
    h_image[i] = i % 256;
  }

  uint64_t hash = dct.dct_host(h_image.data());

  std::cout << "Hash: 0x" << std::hex << hash << std::dec << std::endl;
}

// Example 4: Batch processing with default stream
void example_batch() {
  std::cout << "\n=== Example 4: Batch Processing ===" << std::endl;

  const int N = 32;
  const int BATCH_SIZE = 10;

  GpuDct<float> dct(N);

  // Create batch of test images
  std::vector<float> h_images(N * N * BATCH_SIZE);
  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int i = 0; i < N * N; i++) {
      h_images[b * N * N + i] = static_cast<float>((i + b * 10) % 256);
    }
  }

  std::vector<uint64_t> h_hashes(BATCH_SIZE);

  auto start = std::chrono::high_resolution_clock::now();
  dct.batch_dct_host(h_images.data(), h_hashes.data(), BATCH_SIZE);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Processed " << BATCH_SIZE << " images in " << duration.count()
            << " μs" << std::endl;

  for (int i = 0; i < BATCH_SIZE; i++) {
    std::cout << "Image " << i << " Hash: 0x" << std::hex << h_hashes[i]
              << std::dec << std::endl;
  }
}

// Example 5: Multi-stream parallel processing
void example_multistream() {
  std::cout << "\n=== Example 5: Multi-Stream Parallel Processing ==="
            << std::endl;

  const int N = 32;
  const int BATCH_SIZE = 16;
  const int NUM_STREAMS = 4;

  GpuDct<float> dct(N);

  // Create batch of test images
  std::vector<float> h_images(N * N * BATCH_SIZE);
  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int i = 0; i < N * N; i++) {
      h_images[b * N * N + i] = static_cast<float>((i + b * 10) % 256);
    }
  }

  // Allocate device memory for images and hashes
  float *d_images;
  uint64_t *d_hashes;
  cudaMalloc(&d_images, N * N * BATCH_SIZE * sizeof(float));
  cudaMalloc(&d_hashes, BATCH_SIZE * sizeof(uint64_t));

  // Upload images
  cudaMemcpy(d_images, h_images.data(), N * N * BATCH_SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  // Create streams
  std::vector<cudaStream_t> streams(NUM_STREAMS);
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&streams[i]);
  }

  auto start = std::chrono::high_resolution_clock::now();
  dct.batch_dct_device_multistream(d_images, d_hashes, BATCH_SIZE,
                                   streams.data(), NUM_STREAMS);
  auto end = std::chrono::high_resolution_clock::now();

  // Download results
  std::vector<uint64_t> h_hashes(BATCH_SIZE);
  cudaMemcpy(h_hashes.data(), d_hashes, BATCH_SIZE * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Processed " << BATCH_SIZE << " images with " << NUM_STREAMS
            << " streams in " << duration.count() << " μs" << std::endl;

  for (int i = 0; i < BATCH_SIZE; i++) {
    std::cout << "Image " << i << " Hash: 0x" << std::hex << h_hashes[i]
              << std::dec << std::endl;
  }

  // Cleanup
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(streams[i]);
  }
  cudaFree(d_images);
  cudaFree(d_hashes);
}

// Example 6: Async operations with pinned memory
void example_async() {
  std::cout << "\n=== Example 6: Async Operations with Pinned Memory ==="
            << std::endl;

  const int N = 32;
  GpuDct<float> dct(N);

  // Allocate pinned memory for async transfers
  float *h_image;
  uint64_t *h_hash;
  cudaMallocHost(&h_image, N * N * sizeof(float));
  cudaMallocHost(&h_hash, sizeof(uint64_t));

  // Create test image
  for (int i = 0; i < N * N; i++) {
    h_image[i] = static_cast<float>(i % 256);
  }

  // Create custom stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Launch async operation
  dct.dct_host_async(h_image, h_hash, stream);

  // Can do other work here while GPU is computing...
  std::cout << "GPU is computing asynchronously..." << std::endl;

  // Wait for completion
  cudaStreamSynchronize(stream);

  std::cout << "Hash: 0x" << std::hex << *h_hash << std::dec << std::endl;

  // Cleanup
  cudaStreamDestroy(stream);
  cudaFreeHost(h_image);
  cudaFreeHost(h_hash);
}

// Example 7: Device-to-device (no host involvement)
void example_device_only() {
  std::cout << "\n=== Example 7: Device-to-Device Operations ===" << std::endl;

  const int N = 32;
  GpuDct<float> dct(N);

  // Allocate device memory directly
  float *d_image;
  uint64_t *d_hash;
  cudaMalloc(&d_image, N * N * sizeof(float));
  cudaMalloc(&d_hash, sizeof(uint64_t));

  // Initialize device memory with a kernel or other GPU operation
  // (For demo, we'll just use cudaMemset)
  cudaMemset(d_image, 42, N * N * sizeof(float));

  // Compute DCT hash directly on device (stream-ordered memory allocated
  // internally)
  dct.dct_device(d_image, d_hash);

  // Download result
  uint64_t h_hash;
  cudaMemcpy(&h_hash, d_hash, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  std::cout << "Hash: 0x" << std::hex << h_hash << std::dec << std::endl;

  // Cleanup
  cudaFree(d_image);
  cudaFree(d_hash);
}

int main() {
  std::cout << "GpuDct Library Examples - Stream-Ordered Memory Edition"
            << std::endl;
  std::cout << "========================================================="
            << std::endl;

  example_single_float();
  example_single_double();
  example_integer();
  example_batch();
  example_multistream();
  example_async();
  example_device_only();

  std::cout << "\n========================================================="
            << std::endl;
  std::cout << "All examples completed successfully!" << std::endl;

  return 0;
}
