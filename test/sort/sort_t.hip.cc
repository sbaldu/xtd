
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

#include "common/hip_check.h"
#include <hip_runtime.h>

TEST_CASE("sortHIP", "[sort]") {
  int deviceCount;
  hipError_t hipStatus = hipGetDeviceCount(&deviceCount);

  if (hipStatus != hipSuccess || deviceCount == 0) {
    std::cout << "No AMD GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }
  HIP_CHECK(hipSetDevice(0));

  const int N = 100;
  std::random_device rd;
  std::mt19937 rng(rd());

  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::shuffle(values.begin(), values.end(), rng);
  xtd::sort(values.begin(), values.end());

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int* d_values;
  HIP_CHECK(hipMallocAsync(&d_values, N * sizeof(int), stream));
  HIP_CHECK(hipMemcpyAsync(d_values, values.data(), N * sizeof(int), hipMemcpyHostToDevice, stream));
  xtd::sort(d_values, d_values + N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(values.data(), d_values, N * sizeof(int), hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(std::ranges::equal(values, std::views::iota(N, 0)));

  HIP_CHECK(hipFreeAsync(d_values, stream));
  HIP_CHECK(hipStreamDestroy(stream));
}
