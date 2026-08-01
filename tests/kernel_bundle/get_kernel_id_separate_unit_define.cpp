/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  This translation unit defines (names) the kernel that is later looked up,
//  in a different translation unit, via
//  sycl::get_kernel_id<get_kernel_id_separate_unit::kernel>(). This file
//  intentionally contains no test cases of its own; see
//  get_kernel_id_separate_unit.cpp for the actual test.
//
*******************************************************************************/

#include "get_kernel_id_separate_unit.h"

namespace get_kernel_id_separate_unit {

void submit_kernel(sycl::queue& queue, sycl::buffer<int, 1>& buf) {
  queue.submit([&](sycl::handler& cgh) {
    sycl::accessor acc{buf, cgh, sycl::write_only};
    cgh.single_task<kernel>([=]() { acc[0] = expected_value; });
  });
}

}  // namespace get_kernel_id_separate_unit
