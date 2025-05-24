
#pragma once

#include "xtd/internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T expm1(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::expm1(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::expm1(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::expm1(x);
#else
  // standard C++ code
  return std::expm1(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float expm1f(float x) {
  return expm1(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double expm1l(long double x) {
  return expm1(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double expm1(T x) {
	return expm1(static_cast<double>(x));
}

} // namespace xtd
