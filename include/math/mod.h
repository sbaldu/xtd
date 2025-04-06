
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float fmod(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmod(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmod(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmod(x, y);
#else
    // standard C++ code
    return std::fmod(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double fmod(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmod(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmod(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmod(x, y);
#else
    // standard C++ code
    return std::fmod(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float fmodf(float x, float y) { return fmodf(x, y); }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION inline constexpr double fmod(T x, T y) {
    return fmod(static_cast<double>(x), static_cast<double>(y));
  }

}  // namespace xtd
