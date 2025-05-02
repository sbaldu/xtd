
#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include "algorithm.h"

TEST_CASE("sortCPU", "[sort]") {
  const int N = 100;
  std::vector<int> values(N);
  std::iota(values.begin(), values.end(), 0);
  std::random_shuffle(values.begin(), values.end());

  xtd::sort(values.begin(), values.end());

  REQUIRE_THAT(std::equal(values, std::views::iota(N, 0)));
}
