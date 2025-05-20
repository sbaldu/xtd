
#pragma once

#include "../internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T atan2(T x, T y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan2(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan2(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan2(x, y);
#else
    // standard C++ code
    return std::atan2(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float atan2f(float x, float y) {
    return atan2(x, y);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double atan2l(long double x, long double y) {
    return atan2(x, y);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double atan2(T x, T y) {
    return atan2(static_cast<double>(x), static_cast<double>(y));
  }


}  // namespace xtd
