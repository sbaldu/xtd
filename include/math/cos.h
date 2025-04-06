
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION
  inline constexpr float cos(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(arg);
#else
    // standard C++ code
    return std::cos(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double cos(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(arg);
#else
    // standard C++ code
    return std::cos(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float cosf(float arg) { return cos(arg); }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION inline constexpr double cos(T arg) {
    return cos(static_cast<double>(arg));
  }

}  // namespace xtd
