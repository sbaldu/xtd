
#pragma once

#if defined(XTD_TARGET_CUDA)
#include <thrust/sort.h>
#elif defined(XTD_TARGET_HIP)
#include <rocthrust/sort.h>
#elif defined(XTD_TARGET_SYCL)
#include <oneapi/dpl/algorithm>
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
    oneapi::dpl::sort(sycl::execution::dpcpp_default, first, last);
#else
    std::sort(first, last);
#endif
  }

}  // namespace xtd
