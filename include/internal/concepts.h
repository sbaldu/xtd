
#pragma once

#include <type_traits>

namespace xtd {

  template <typename T>
  concept Numeric = requires {
    std::is_arithmetic_v<T>;
    requires sizeof(T) <= 8;
  };

}  // namespace xtd
