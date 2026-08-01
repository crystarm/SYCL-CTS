/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  SYCL 2020 section 4.11.6, "Obtaining a kernel identifier" requires that
//  the template parameter KernelName passed to sycl::get_kernel_id() name a
//  kernel that is defined somewhere in the SYCL application; it need not be
//  defined in the same translation unit as the call to
//  sycl::get_kernel_id<KernelName>(). This test verifies that
//  sycl::get_kernel_id() succeeds, and returns a kernel_id that can be used
//  to obtain and execute the kernel, when the named kernel is defined only
//  in a different translation unit
//  (get_kernel_id_separate_unit_define.cpp) than the one containing the call
//  to sycl::get_kernel_id() (this file).
//
//  See: https://github.com/KhronosGroup/SYCL-CTS/issues/1212
//
*******************************************************************************/

#include "../common/common.h"
#include "get_kernel_id_separate_unit.h"

namespace get_kernel_id_separate_unit {

TEST_CASE(
    "sycl::get_kernel_id() succeeds for a kernel defined in a different "
    "translation unit",
    "[kernel_id][kernel_bundle]") {
  auto queue = sycl_cts::util::get_cts_object::queue();
  auto ctx = queue.get_context();

  // The `kernel` class is only ever named in a kernel invocation function
  // (handler::single_task()) in get_kernel_id_separate_unit_define.cpp; it is
  // otherwise an incomplete type in this translation unit. It must still be
  // possible to call sycl::get_kernel_id() for it here, since that other
  // translation unit is linked into this same test executable.
  sycl::kernel_id k_id = sycl::get_kernel_id<kernel>();
  CHECK(k_id.get_name() != nullptr);

  // Verify that a kernel_bundle containing the kernel can be obtained using
  // the kernel_id retrieved above, and that the kernel bundle reports the
  // kernel as present.
  auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, ctx.get_devices(), {k_id});
  CHECK(kb.has_kernel(k_id));

  // Verify that the kernel, defined in the other translation unit, can
  // actually be executed and produces the expected result.
  int result = 0;
  {
    sycl::buffer<int, 1> buf{&result, sycl::range<1>(1)};
    submit_kernel(queue, buf);
  }
  queue.wait_and_throw();
  CHECK(result == expected_value);
}

}  // namespace get_kernel_id_separate_unit
