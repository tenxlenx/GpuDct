#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gpu_dct.cuh>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace {

struct ImageData {
    int width{0};
    int height{0};
    std::vector<unsigned char> pixels;
};

ImageData load_grayscale_image(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Image file not found: " + path.string());
    }

    int width = 0;
    int height = 0;
    int components = 0;
    unsigned char *data = stbi_load(path.string().c_str(), &width, &height,
                                    &components, STBI_grey);
    if (data == nullptr) {
        throw std::runtime_error("Failed to load image: " + path.string());
    }

    ImageData image;
    image.width = width;
    image.height = height;
    image.pixels.assign(data, data + (static_cast<size_t>(width) * height));
    stbi_image_free(data);
    return image;
}

std::vector<float> resample_to_size(const ImageData &image, int target) {
    if (image.width < target || image.height < target) {
        throw std::runtime_error(
            "Image dimensions are smaller than target size");
    }

    std::vector<float> output(static_cast<size_t>(target) * target);
    const float x_scale = static_cast<float>(image.width) / target;
    const float y_scale = static_cast<float>(image.height) / target;

    for (int y = 0; y < target; ++y) {
        const int src_y =
            std::min(static_cast<int>(y * y_scale), image.height - 1);
        for (int x = 0; x < target; ++x) {
            const int src_x =
                std::min(static_cast<int>(x * x_scale), image.width - 1);
            output[static_cast<size_t>(y) * target + x] = static_cast<float>(
                image.pixels[static_cast<size_t>(src_y) * image.width + src_x]);
        }
    }

    return output;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const std::filesystem::path image_path =
            (argc > 1) ? argv[1]
                       : std::filesystem::path("examples/data/lena.jpg");
        const int size = (argc > 2) ? std::stoi(argv[2]) : 256;

        static constexpr int kSupportedSizes[] = {32, 64, 128, 256};
        if (std::find(std::begin(kSupportedSizes), std::end(kSupportedSizes),
                      size) == std::end(kSupportedSizes)) {
            throw std::runtime_error(
                "Supported DCT sizes are 32, 64, 128, 256");
        }

        std::cout << "Loading image: " << image_path << "\n";
        auto image = load_grayscale_image(image_path);
        std::cout << "Original dimensions: " << image.width << "x"
                  << image.height << "\n";

        auto resampled = resample_to_size(image, size);

        gpu_dct::GpuDct<float> dct(size);
        const uint64_t hash = dct.dct_host(resampled.data());

        std::cout << "Computed hash (" << size << "x" << size << "): 0x"
                  << std::hex << hash << std::dec << "\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
