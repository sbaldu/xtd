
#pragma once

#include "internal/defines.h"

#if defined(XTD_TARGET_CUDA)
#include <thrust/sort.h>
#elif defined(XTD_TARGET_HIP)
#include <rocthrust/sort.h>
#elif defined(XTD_TARGET_SYCL)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace xtd {

  template <typename RandomAccessIterator>
  XTD_DEVICE_FUNCTION inline constexpr void sort(RandomAccessIterator first,
                                                 RandomAccessIterator last) {
#if defined(XTD_TARGET_CUDA)
    thrust::sort(first, last);
#elif defined(XTD_TARGET_HIP)
    rocthrust::sort(first, last);
#elif defined(XTD_TARGET_SYCL)
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    std::sort(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator>
  XTD_DEVICE_FUNCTION inline constexpr void sort(ExecutionPolicy&& policy,
                                                 RandomAccessIterator first,
                                                 RandomAccessIterator last) {
#if defined(XTD_TARGET_CUDA)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_HIP)
    rocthrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_SYCL)
    oneapi::dpl::sort(std::forward<ExecutionPolicy>(policy), first, last);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename RandomAccessIterator, typename Compare>
  XTD_DEVICE_FUNCTION inline constexpr void sort(RandomAccessIterator first,
                                                 RandomAccessIterator last,
                                                 Compare comp) {
#if defined(XTD_TARGET_CUDA)
    thrust::sort(first, last, comp);
#elif defined(XTD_TARGET_HIP)
    rocthrust::sort(first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    std::sort(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
  XTD_DEVICE_FUNCTION inline constexpr void sort(ExecutionPolicy&& policy,
                                                 RandomAccessIterator first,
                                                 RandomAccessIterator last,
                                                 Compare comp) {
#if defined(XTD_TARGET_CUDA)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_HIP)
    rocthrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    oneapi::dpl::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

}  // namespace xtd
