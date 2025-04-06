/*
 * Copyright 2024 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  /* Computes the sine of arg (measured in radians),
   * in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sin(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(arg);
#else
    // standard C++ code
    return std::sin(arg);
#endif
  }

  /* Computes the sine of arg (measured in radians),
   * in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sin(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(arg);
#else
    // standard C++ code
    return std::sin(arg);
#endif
  }

  /* Computes the sine of arg (measured in radians),
   * in double precision.
   */
  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double sin(T arg) {
    return sin(static_cast<double>(arg));
  }

  /* Computes the sine of arg (measured in radians),
   * in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sinf(float arg) { return sin(arg); }

  /* Computes the sine of arg (measured in radians),
   * in single precision.
   */
  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double sinf(T arg) {
    return sin(static_cast<float>(arg));
  }

}  // namespace xtd
