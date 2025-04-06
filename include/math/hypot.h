
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float hypot(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::hypot(x, y);
#else
    // standard C++ code
    return std::hypot(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double hypot(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::hypot(x, y);
#else
    // standard C++ code
    return std::hypot(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float hypotf(float x, float y) { return hypot(x, y); }

}  // namespace xtd
