
#pragma once

#include "xtd/internal/defines.h"
#include "xtd/internal/concepts.h"

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <Numeric T>
  XTD_DEVICE_FUNCTION inline constexpr const T& max(const T& a, const T& b) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::max(a, b);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::max(a, b);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::max(a, b);
#else
    // standard C++ code
    return std::max(a, b);
#endif
  }

  template <Numeric T, typename Compare>
  XTD_DEVICE_FUNCTION inline constexpr const T& max(const T& a, const T& b, Compare comp) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::max(a, b, comp);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::max(a, b, comp);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::max(a, b, comp);
#else
    // standard C++ code
    return std::max(a, b, comp);
#endif
  }

}  // namespace xtd
