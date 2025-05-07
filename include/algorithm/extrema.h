
#pragma once

#include "internal/defines.h"

#if defined(XTD_TARGET_CUDA)
#include <thrust/extrema.h>
#elif defined(XTD_TARGET_HIP)
#include <rocthrust/extrema.h>
#elif defined(XTD_TARGET_SYCL)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace xtd {

  template <typename ForwardIterator>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator min_element(ForwardIterator first,
                                                                   ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::min_element(first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::min_element(first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::min_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::min_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator min_element(ExecutionPolicy&& policy,
                                                                   ForwardIterator first,
                                                                   ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::min_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator min_element(ForwardIterator first,
                                                                   ForwardIterator last,
                                                                   BinaryPredicate comp) {
#if defined(XTD_TARGET_CUDA)
    return thrust::min_element(first, last, comp);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::min_element(first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::min_element(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    return std::min_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator min_element(ExecutionPolicy&& policy,
                                                                   ForwardIterator first,
                                                                   ForwardIterator last,
                                                                   BinaryPredicate comp) {
#if defined(XTD_TARGET_CUDA)
    return thrust::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::min_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <typename ForwardIterator>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator max_element(ForwardIterator first,
                                                                   ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::max_element(first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::max_element(first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::max_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::max_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator max_element(ExecutionPolicy&& policy,
                                                                   ForwardIterator first,
                                                                   ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::max_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator max_element(ForwardIterator first,
                                                                   ForwardIterator last,
                                                                   BinaryPredicate comp) {
#if defined(XTD_TARGET_CUDA)
    return thrust::max_element(first, last, comp);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::max_element(first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::max_element(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    return std::max_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  XTD_DEVICE_FUNCTION inline constexpr ForwardIterator max_element(ExecutionPolicy&& policy,
                                                                   ForwardIterator first,
                                                                   ForwardIterator last,
                                                                   BinaryPredicate comp) {
#if defined(XTD_TARGET_CUDA)
    return thrust::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::max_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

  template <typename ForwardIterator>
  XTD_DEVICE_FUNCTION inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ForwardIterator first, ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::minmax_element(first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::minmax_element(first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::minmax_element(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::minmax_element(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  XTD_DEVICE_FUNCTION inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::minmax_element(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename ForwardIterator, typename BinaryPredicate>
  XTD_DEVICE_FUNCTION inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ForwardIterator first, ForwardIterator last, BinaryPredicate comp) {
#if defined(XTD_TARGET_CUDA)
    return thrust::minmax_element(first, last, comp);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::minmax_element(first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::minmax_element(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    return std::minmax_element(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  XTD_DEVICE_FUNCTION inline constexpr std::pair<ForwardIterator, ForwardIterator> minmax_element(
      ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last, BinaryPredicate comp) {
#if defined(XTD_TARGET_CUDA)
    return thrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    return std::minmax_element(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

}  // namespace xtd
