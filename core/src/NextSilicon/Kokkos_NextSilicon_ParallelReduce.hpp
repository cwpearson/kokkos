// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_REDUCE_HPP

#include <type_traits>
#include <cstdint>
#include <mutex>
#include <Kokkos_Parallel.hpp>
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Intrinsics.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ThreadSpaceGuard.hpp>
#include <nextapi/parallelism.hpp>
#include <nextapi/memory.h>

namespace Kokkos::Impl {

template <class CombinedFunctorReducerType, class... Traits>
struct NextSiliconParallelReduceImpl {
  using ExecSpace   = Kokkos::Experimental::NextSilicon;
  using MemorySpace = typename ExecSpace::memory_space;
  using Policy      = RangePolicy<Traits...>;
  using IndexType   = typename Policy::index_type;
  using WorkRange   = typename Policy::WorkRange;
  using WorkTag     = typename Policy::work_tag;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using Pointer       = typename ReducerType::pointer_type;
  using ValueType     = typename ReducerType::value_type;
  using ReferenceType = typename ReducerType::reference_type;

  // Each thread does part of the input array
  // FIXME_NEXTSILICON: make this user-controllable
  constexpr static uint32_t MAX_PARTIAL_PROD = 16 * 1024;  // 16K

  // Let threads reduce the problem by at least a factor of
  // MIN_ITER_PER_PARTIAL_PROD in case of smaller inputs.
  // FIXME_NEXTSILICON: make this user-controllable
  constexpr static uint32_t MIN_ITER_PER_PARTIAL_PROD = 100;

  constexpr static uint32_t TREE_JOIN_THRESHOLD = 128;

  constexpr static uint32_t SCRATCH_ALLOC_ALIGNMENT = 64;

  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  Pointer m_result_ptr;

  NextSiliconParallelReduceImpl(
      CombinedFunctorReducerType const& functor_reducer, Policy const& policy,
      Pointer result_ptr)
      : m_functor_reducer(functor_reducer),
        m_policy(policy),
        m_result_ptr(result_ptr) {}

  void execute() const {
    // Acquire the device for potential handoff before kernel execution begins
    const std::lock_guard<std::recursive_mutex> device_lock =
        this->m_policy.space().impl_internal_space_instance()->lock_device();

    IndexType begin = m_policy.begin();
    IndexType end   = m_policy.end();

    const auto& reducer      = m_functor_reducer.get_reducer();
    IndexType num_iterations = end - begin;

    if (end <= begin) {
      num_iterations = 0;
    }

    // Calculate optimal number of partial products
    uint32_t num_partial_prod =
        (num_iterations + (MIN_ITER_PER_PARTIAL_PROD - 1)) /
        (MIN_ITER_PER_PARTIAL_PROD);
    num_partial_prod = std::min(num_partial_prod, MAX_PARTIAL_PROD);

    // Even if num_iterations 0, set num_partial_prod to at least 1 to handle
    // the case where we need to init/final the result_ptr.
    if (num_partial_prod == 0) num_partial_prod = 1;

    const auto value_count = reducer.value_count();
    const auto values_size = value_count * sizeof(ValueType);

    // Round up to SCRATCH_ALLOC_ALIGNMENT
    size_t result_arr_size =
        ((values_size * num_partial_prod + SCRATCH_ALLOC_ALIGNMENT - 1) /
         SCRATCH_ALLOC_ALIGNMENT) *
        SCRATCH_ALLOC_ALIGNMENT;

    auto internal_instance = m_policy.space().impl_internal_space_instance();

    std::byte* partial_buffer =
        internal_instance->resize_reduce_partial_buffer(result_arr_size);

    ValueType* const result_arr = reinterpret_cast<ValueType*>(partial_buffer);

    // Clone the driver before going into the handoff function.
    // This prevents the stack from getting migrated into device.
    auto cloned_driver = internal_instance->clone_driver(*this);
    cloned_driver->execute_internal(num_partial_prod, result_arr);

    // Final result is at the begining of values array
    // FIXME_NEXTSILICON: if m_result_ptr is not 64B aligned,
    // nextapi_memory_copy may cause m_result_ptr and result_arr to migrate to
    // host
    if (m_result_ptr) {
      nextapi_memory_copy(m_result_ptr, result_arr,
                          sizeof(ValueType) * reducer.value_count());
    }
  }

  void execute_internal(uint32_t num_partial_prod,
                        ValueType* const result_arr) const {
    IndexType chunk_size = 1;

    // allow init/final-only case (empty iteration space) to be handed off
    if (m_policy.end() <= m_policy.begin()) {
      execute_init_final(chunk_size, result_arr);
      return;
    }

    execute_partial_reduce(num_partial_prod, chunk_size, result_arr);

    uint32_t final_join_count  = num_partial_prod;
    uint32_t final_join_stride = 1;

    // if there are lots of partial products, allow parallel joins of groups of
    // final_join_stride to be handed off
    if (num_partial_prod > TREE_JOIN_THRESHOLD) {
      final_join_stride = TREE_JOIN_THRESHOLD;
      final_join_count =
          (num_partial_prod + final_join_stride - 1) / final_join_stride;

      execute_tree_join(num_partial_prod, final_join_count, final_join_stride,
                        chunk_size, result_arr);
    }

    // allow final sequential join to be handed off
    execute_final_join(final_join_count, final_join_stride, chunk_size,
                       result_arr);
  }

  __attribute__((noinline)) void execute_init_final(
      IndexType chunk_size, ValueType* const result_arr) const {
    const auto& reducer = m_functor_reducer.get_reducer();

    nextapi::parallel_for<nextapi::work_distribution::STATIC>(
        IndexType(0), IndexType(1), {.chunk_size = chunk_size},
        init_final_function, &reducer, result_arr);
  }

  __attribute__((noinline)) void execute_partial_reduce(
      uint32_t num_partial_prod, IndexType chunk_size,
      ValueType* const result_arr) const {
    const auto& functor = m_functor_reducer.get_functor();
    const auto& reducer = m_functor_reducer.get_reducer();

    nextapi::parallel_for<nextapi::work_distribution::STATIC>(
        IndexType(0), static_cast<IndexType>(num_partial_prod),
        {.chunk_size = chunk_size}, parallel_function, num_partial_prod,
        &functor, &reducer, &m_policy, result_arr);
  }

  __attribute__((noinline)) void execute_tree_join(
      uint32_t num_partial_prod, uint32_t final_join_count,
      uint32_t final_join_stride, IndexType chunk_size,
      ValueType* const result_arr) const {
    const auto& reducer = m_functor_reducer.get_reducer();

    nextapi::parallel_for<nextapi::work_distribution::STATIC>(
        IndexType(0), static_cast<IndexType>(final_join_count),
        {.chunk_size = chunk_size}, tree_join_function, num_partial_prod,
        final_join_stride, &reducer, result_arr);
  }

  __attribute__((noinline)) void execute_final_join(
      uint32_t final_join_count, uint32_t final_join_stride,
      IndexType chunk_size, ValueType* const result_arr) const {
    const auto& reducer = m_functor_reducer.get_reducer();

    nextapi::parallel_for<nextapi::work_distribution::STATIC>(
        IndexType(0), IndexType(1), {.chunk_size = chunk_size},
        final_join_function, final_join_count, final_join_stride, &reducer,
        result_arr);
  }

  static void init_final_function(IndexType /*thread_index*/,
                                  const ReducerType* reducer,
                                  ValueType* result_arr) {
    // Set the thread's semantic execution space to device
    //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed description
    //   of the mechanism and rationale.
    NextSiliconThreadSpaceGuard thread_guard{};

    if (__next_is_in_handed_off_code()) {
      nextapi::detail::__next_immutable_thread_invariant_parameter_struct(
          reducer);
    }

    reducer->init(&result_arr[0]);
    reducer->final(&result_arr[0]);
  }

  static void parallel_function(IndexType thread_index,
                                uint32_t num_partial_prod,
                                const FunctorType* functor,
                                const ReducerType* reducer,
                                const Policy* policy, ValueType* result_arr) {
    // Set the thread's semantic execution space to device
    //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed description
    //   of the mechanism and rationale.
    NextSiliconThreadSpaceGuard thread_guard{};

    // Communicate to the compiler that the functor is immutable and thread
    // invariant for the duration of the microtask.
    if (__next_is_in_handed_off_code()) {
      nextapi::detail::__next_immutable_thread_invariant_parameter_struct(
          functor);
    }

    WorkRange range(*policy, thread_index, num_partial_prod);

    auto value_count    = reducer->value_count();
    auto* interm_values = &result_arr[thread_index * value_count];

    reducer->init(interm_values);

    ReferenceType ref = reducer->reference(interm_values);

    for (auto i = range.begin(); i < range.end(); ++i) {
      if constexpr (std::is_void_v<WorkTag>) {
        (*functor)(i, ref);
      } else {
        (*functor)(WorkTag{}, i, ref);
      }
    }
  }

  static void tree_join_function(IndexType group_index,
                                 uint32_t num_partial_prod, uint32_t stride,
                                 const ReducerType* reducer,
                                 ValueType* result_arr) {
    // Set the thread's semantic execution space to device
    //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed description
    //   of the mechanism and rationale.
    NextSiliconThreadSpaceGuard thread_guard{};

    if (__next_is_in_handed_off_code()) {
      nextapi::detail::__next_immutable_thread_invariant_parameter_struct(
          reducer);
    }

    const auto value_count   = reducer->value_count();
    const uint32_t group_beg = static_cast<uint32_t>(group_index) * stride;
    const uint32_t group_end = std::min(group_beg + stride, num_partial_prod);
    ValueType* group_acc     = &result_arr[group_beg * value_count];

    for (uint32_t i = group_beg + 1; i < group_end; ++i) {
      reducer->join(group_acc, &result_arr[i * value_count]);
    }
  }

  static void final_join_function(IndexType /*thread_index*/,
                                  uint32_t num_elements, uint32_t stride,
                                  const ReducerType* reducer,
                                  ValueType* result_arr) {
    // Set the thread's semantic execution space to device
    //   See Kokkos_NextSilicon_ThreadSpaceGuard.hpp for a detailed description
    //   of the mechanism and rationale.
    NextSiliconThreadSpaceGuard thread_guard{};

    if (__next_is_in_handed_off_code()) {
      nextapi::detail::__next_immutable_thread_invariant_parameter_struct(
          reducer);
    }

    const auto value_count = reducer->value_count();

    for (uint32_t i = 1; i < num_elements; ++i) {
      reducer->join(&result_arr[0], &result_arr[i * stride * value_count]);
    }

    reducer->final(&result_arr[0]);
  }
};

}  // namespace Kokkos::Impl

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_HPP */
