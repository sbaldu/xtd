
#pragma once

#include "xtd/internal/defines.h"

#if defined(XTD_TARGET_CUDA)
#include <thrust/reduce.h>
#elif defined(XTD_TARGET_HIP)
#include <rocthrust/reduce.h>
#elif defined(XTD_TARGET_SYCL)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace xtd {

  template <typename InputIterator>
  XTD_HOST_FUNCTION inline constexpr typename std::iterator_traits<InputIterator>::value_type
  reduce(InputIterator first, InputIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::reduce(thrust::device, first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::reduce(thrust::hip::par, first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::reduce(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    return std::reduce(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator>
  XTD_HOST_FUNCTION inline constexpr typename std::iterator_traits<ForwardIterator>::value_type
  reduce(ExecutionPolicy&& policy, ForwardIterator first, ForwardIterator last) {
#if defined(XTD_TARGET_CUDA)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename InputIterator, typename T>
  XTD_HOST_FUNCTION inline constexpr T reduce(InputIterator first, InputIterator last, T init) {
#if defined(XTD_TARGET_CUDA)
    return thrust::reduce(thrust::device, first, last, init);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::reduce(thrust::hip::par, first, last, init);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::reduce(oneapi::dpl::execution::dpcpp_default, first, last, init);
#else
    return std::reduce(first, last, init);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  XTD_HOST_FUNCTION inline constexpr T reduce(ExecutionPolicy&& policy,
                                              ForwardIterator first,
                                              ForwardIterator last,
                                              T init) {
#if defined(XTD_TARGET_CUDA)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last, init);
#endif
  }

  template <typename InputIterator, typename T, typename BinaryOperation>
  XTD_HOST_FUNCTION inline constexpr T reduce(InputIterator first,
                                              InputIterator last,
                                              T init,
                                              BinaryOperation op) {
#if defined(XTD_TARGET_CUDA)
    return thrust::reduce(thrust::device, first, last, init, op);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::reduce(thrust::hip::par, first, last, init, op);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::reduce(oneapi::dpl::execution::dpcpp_default, first, last, init, op);
#else
    return std::reduce(first, last, init, op);
#endif
  }

  template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryOperation>
  XTD_HOST_FUNCTION inline constexpr T reduce(ExecutionPolicy&& policy,
                                              ForwardIterator first,
                                              ForwardIterator last,
                                              T init,
                                              BinaryOperation op) {
#if defined(XTD_TARGET_CUDA)
    return thrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#elif defined(XTD_TARGET_HIP)
    return rocthrust::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#elif defined(XTD_TARGET_SYCL)
    return oneapi::dpl::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#else
    return std::reduce(std::forward<ExecutionPolicy>(policy), first, last, init, op);
#endif
  }

}  // namespace xtd
