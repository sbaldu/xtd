
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include "common/cuda_check.h"
#include <cuda_runtime.h>

TEST_CASE("sortCUDA", "[sort]") {
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }
  CUDA_CHECK(cudaSetDevice(0));

  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);
  xtd::sort(values.begin(), values.end());

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  int* d_values;
  CUDA_CHECK(cudaMallocAsync(&d_values, N * sizeof(int), stream));
  CUDA_CHECK(cudaMemcpyAsync(d_values, values.data(), N * sizeof(int), cudaMemcpyHostToDevice, stream));
  xtd::sort(d_values, d_values + N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(values.data(), d_values, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  REQUIRE(std::ranges::equal(values, std::views::iota(N, 0)));

  CUDA_CHECK(cudaFreeAsync(d_values, stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}
