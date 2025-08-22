/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <vector>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), ResultType (*StdFunc)(InputType)>
void test(std::vector<double> const& values) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the std reference
    ResultType reference = StdFunc(input);
    CHECK_THAT(result, Catch::Matchers::WithinULP(reference, 1));
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), ResultType (*StdFunc)(float)>
void test_f(std::vector<double> const& values) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the std reference
    ResultType reference = StdFunc(static_cast<float>(input));
    CHECK_THAT(result, Catch::Matchers::WithinULP(reference, 1));
  }
}
