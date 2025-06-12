
#pragma once

#include "xtd/internal/defines.h"

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float pow(float base, float exp) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::pow(base, exp);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::pow(base, exp);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::pow(base, exp);
#else
    // standard C++ code
    return std::pow(base, exp);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double pow(double base, double exp) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::pow(base, exp);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::pow(base, exp);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::pow(base, exp);
#else
    // standard C++ code
    return std::pow(base, exp);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float powf(float base, float exp) { return pow(base, exp); }

}  // namespace xtd
