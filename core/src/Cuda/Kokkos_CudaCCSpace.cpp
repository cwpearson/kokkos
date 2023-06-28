//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_CUDA

#include <Kokkos_Core.hpp>
#include <Cuda/Kokkos_Cuda.hpp>
#include <Cuda/Kokkos_CudaCCSpace.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>

//#include <Cuda/Kokkos_Cuda_BlockSize_Deduction.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_MemorySpace.hpp>

#include <impl/Kokkos_Tools.hpp>


namespace Kokkos {

CudaCCSpace::CudaCCSpace() : m_device(Kokkos::Cuda().cuda_device()) {}

void *CudaCCSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *CudaCCSpace::allocate(const char *arg_label, const size_t arg_alloc_size,
                          const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}


namespace {
void *impl_allocate_common(const Cuda &exec_space, const char *arg_label,
                           const size_t arg_alloc_size,
                           const size_t arg_logical_size,
                           const Kokkos::Tools::SpaceHandle arg_handle,
                           bool exec_space_provided) {
  (void)exec_space;
  (void)exec_space_provided;
  (void)arg_logical_size;
  void *ptr = malloc(arg_alloc_size);
  if (arg_alloc_size && !ptr) {

    throw Experimental::RawMemoryAllocationFailure(
      arg_alloc_size, 0/*attempted alignment? FIXME_CUDACCSPACE */
      );

  }



  return ptr;
}
}  // namespace




void *CudaCCSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  return impl_allocate_common(Kokkos::Cuda{}, arg_label, arg_alloc_size,
                              arg_logical_size, arg_handle, false);
}


#if 0
void *CudaCCSpace::impl_allocate(
    const Cuda &exec_space, const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  return impl_allocate_common(exec_space, arg_label, arg_alloc_size,
                              arg_logical_size, arg_handle, true);
}
#endif



void CudaCCSpace::deallocate(void *const arg_alloc_ptr,
                           const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}
void CudaCCSpace::deallocate(const char *arg_label, void *const arg_alloc_ptr,
                           const size_t arg_alloc_size,
                           const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}
void CudaCCSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }
  Impl::cuda_device_synchronize(
        "Kokkos::Cuda: backend fence before async free");
      free(arg_alloc_ptr);
  Impl::cuda_device_synchronize(
        "Kokkos::Cuda: backend fence after async free");
}

} // namespace Kokkos
#if 0

#endif
#endif
