
#pragma once

#include "xtd/internal/defines.h"
#include "xtd/internal/concepts.h"

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <algorithm>
#endif

namespace xtd {

  template <Numeric T>
  XTD_DEVICE_FUNCTION inline constexpr const T& min(const T& a, const T& b) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::min(a, b);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::min(a, b);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::min(a, b);
#else
    // standard C++ code
    return std::min(a, b);
#endif
  }

  template <Numeric T, typename Compare>
  XTD_DEVICE_FUNCTION inline constexpr const T& min(const T& a, const T& b, Compare comp) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::min(a, b, comp);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::min(a, b, comp);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::min(a, b, comp);
#else
    // standard C++ code
    return std::min(a, b, comp);
#endif
  }

}  // namespace xtd
