
#pragma once

#include "xtd/internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION
  inline constexpr float atan(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(arg);
#else
    // standard C++ code
    return std::atan(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double atan(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(arg);
#else
    // standard C++ code
    return std::atan(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float atanf(float arg) { return atan(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double atan(T arg) {
    return atan(static_cast<double>(arg));
  }

}  // namespace xtd
