
#pragma once

#include "internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float cbrt(float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrt(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrt(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(x);
#else
    // standard C++ code
    return std::cbrt(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double cbrt(double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrt(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrt(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(x);
#else
    // standard C++ code
    return std::cbrt(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(float x) { return cbrt(x); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(T x) {
    return cbrt(static_cast<double>(x));
  }

}  // namespace xtd
