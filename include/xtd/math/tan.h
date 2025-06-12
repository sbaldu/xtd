
#pragma once

#include "xtd/internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION
  inline constexpr float tan(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C++ code
    return std::tan(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double tan(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C++ code
    return std::tan(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float tanf(float arg) { return tan(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double tan(T arg) {
    return tan(static_cast<double>(arg));
  }

}  // namespace xtd
