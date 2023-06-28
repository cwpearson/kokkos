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

#ifndef KOKKOS_CUDACC_SHARED_ALLOCATION_RECORD_HPP
#define KOKKOS_CUDACC_SHARED_ALLOCATION_RECORD_HPP

#include <Cuda/Kokkos_CudaCCSpace.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>


template <>
class Kokkos::Impl::SharedAllocationRecord<Kokkos::CudaCCSpace, void>
    : public SharedAllocationRecordCommon<Kokkos::CudaCCSpace> {
 private:
  friend class SharedAllocationRecordCommon<Kokkos::CudaCCSpace>;

  using base_t     = SharedAllocationRecordCommon<Kokkos::CudaCCSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static RecordBase s_root_record;

  const Kokkos::CudaCCSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  // This constructor does not forward to the one without exec_space arg
  // in order to work around https://github.com/kokkos/kokkos/issues/5258
  // This constructor is templated so I can't just put it into the cpp file
  // like the other constructor.
  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/,
      const Kokkos::CudaCCSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate)
      : base_t(
            Impl::checked_allocation_with_header(arg_space, arg_label,
                                                 arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc,
            arg_label),
        m_space(arg_space) {
    this->base_t::_fill_host_accessible_header_info(*base_t::m_alloc_ptr,
                                                    arg_label);
  }

  SharedAllocationRecord(
      const Kokkos::CudaCCSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &base_t::deallocate);
};

#endif
