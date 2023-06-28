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
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_CUDACCSPACE_HPP
#define KOKKOS_CUDACCSPACE_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_CUDA)

#include <Kokkos_Core_fwd.hpp>

#include <iosfwd>
#include <typeinfo>
#include <string>
#include <memory>

#include <Kokkos_HostSpace.hpp>
#include <Cuda/Kokkos_CudaSpace.hpp> // is_cuda_type_space
#include <impl/Kokkos_SharedAlloc.hpp>

#include <impl/Kokkos_Profiling_Interface.hpp>

#include <Cuda/Kokkos_Cuda_abort.hpp>

namespace Kokkos {

class CudaCCSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space    = CudaCCSpace;
  using execution_space = Cuda;
  using device_type     = Kokkos::Device<execution_space, memory_space>;
  using size_type       = unsigned int;

  CudaCCSpace();
  CudaCCSpace(CudaCCSpace&& rhs)      = default;
  CudaCCSpace(const CudaCCSpace& rhs) = default;
  CudaCCSpace& operator=(CudaCCSpace&& rhs) = default;
  CudaCCSpace& operator=(const CudaCCSpace& rhs) = default;
  ~CudaCCSpace()                                  = default;

  /**\brief  Allocate untracked memory in the cuda space */
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

 private:
  template <class, class, class, class>
  friend class Kokkos::Experimental::LogicalMemorySpace;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;

 public:
  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

 private:
  int m_device;  ///< Which Cuda device

  static constexpr const char* m_name = "CudaCC";
};

template <>
struct Impl::is_cuda_type_space<CudaCCSpace> : public std::true_type {};

}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::CudaCCSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, Kokkos::CudaCCSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

// FIXME_CUDACC
// When ENABLE_CUDA_CC and ENABLE_CUDA_UVM, Cuda::memory_space is still cuda UVM
// CudaCCSpace::execution_space::memory_space is CudaUVMSpace, not CudaCCSpace
// some ViewTraits static asserts want CudaCCSpace::execution_space to be able to access CudaCCSpace::memory_space
template <>
struct MemorySpaceAccess<Kokkos::CudaUVMSpace, Kokkos::CudaCCSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaCCSpace, Kokkos::HostSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaCCSpace, Kokkos::CudaSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaCCSpace, Kokkos::CudaUVMSpace> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

}  // namespace Impl
}  // namespace Kokkos

#endif /* #if defined( KOKKOS_ENABLE_CUDA ) */
#endif /* #define KOKKOS_CUDACCSPACE_HPP */
