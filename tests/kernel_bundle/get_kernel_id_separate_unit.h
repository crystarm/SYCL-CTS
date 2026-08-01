/*******************************************************************************
//
//  SPDX-FileCopyrightText: 2026 The Khronos Group Inc.
//  SPDX-License-Identifier: Apache-2.0
//
//  SYCL 2020 Conformance Test Suite
//
//  Provides common declarations shared between two translation units used to
//  verify that sycl::get_kernel_id<KernelName>() succeeds for a KernelName
//  that is defined (i.e. used to name a kernel in a kernel invocation
//  function such as handler::single_task()) in a translation unit other than
//  the one calling sycl::get_kernel_id<KernelName>(). See:
//  https://github.com/KhronosGroup/SYCL-CTS/issues/1212
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_KERNEL_BUNDLE_GET_KERNEL_ID_SEPARATE_UNIT_H
#define __SYCLCTS_TESTS_KERNEL_BUNDLE_GET_KERNEL_ID_SEPARATE_UNIT_H

#include <sycl/sycl.hpp>

namespace get_kernel_id_separate_unit {

/** @brief Value written into the result buffer by the kernel defined in
 *         get_kernel_id_separate_unit_define.cpp
 */
constexpr int expected_value = 42;

/** @brief Name of the kernel that is defined (i.e. named in a call to a
 *         kernel invocation function) in
 *         get_kernel_id_separate_unit_define.cpp, but for which
 *         sycl::get_kernel_id() is only ever called from
 *         get_kernel_id_separate_unit.cpp. This class is intentionally left
 *         incomplete here: only its type identity is needed to name the
 *         kernel and to call sycl::get_kernel_id().
 */
class kernel;

/** @brief Defined in get_kernel_id_separate_unit_define.cpp. Submits the
 *         kernel named by the `kernel` class above, thereby defining it for
 *         the SYCL application. `buf` is written with `expected_value` by
 *         the kernel.
 */
void submit_kernel(sycl::queue& queue, sycl::buffer<int, 1>& buf);

}  // namespace get_kernel_id_separate_unit

#endif  // __SYCLCTS_TESTS_KERNEL_BUNDLE_GET_KERNEL_ID_SEPARATE_UNIT_H
