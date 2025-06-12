
#pragma once

#include "xtd/internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float log(float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log(x);
#else
    // standard C++ code
    return std::log(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double log(double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log(x);
#else
    // standard C++ code
    return std::log(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float logf(float x) { return log(x); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double log(T x) {
    return log(static_cast<double>(x));
  }

}  // namespace xtd
