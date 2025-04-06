
#pragma once

#include "internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float ceil(float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::ceil(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::ceil(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(x);
#else
    // standard C++ code
    return std::ceil(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double ceil(double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::ceil(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::ceil(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(x);
#else
    // standard C++ code
    return std::ceil(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float ceilf(float x) { return ceil(x); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double ceil(T x) {
    return ceil(static_cast<double>(x));
  }

}  // namespace xtd
