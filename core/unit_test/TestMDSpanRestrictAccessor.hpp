// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif
#include <type_traits>

#include <gtest/gtest.h>
#include <concepts>

template <class T, class ExecutionSpace>
void test_restrict_accessor() {
  using value_type = std::remove_const_t<T>;
  Kokkos::View<value_type*, ExecutionSpace> v("V", 100);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(int i) { v(i) = i; });

  int errors;
  using acc_t = Kokkos::Impl::RestrictAccessor<T>;
  acc_t acc{};
  typename acc_t::data_handle_type ptr = v.data();

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(0, v.extent(0)),
      KOKKOS_LAMBDA(int i, int& error) {
        if (acc.access(ptr, i) != ptr[i]) error++;
        if (acc.offset(ptr, i) != ptr + i) error++;
        static_assert(std::is_same_v<typename acc_t::element_type, T>);
        static_assert(std::is_same_v<typename acc_t::reference, T&>);
        static_assert(std::is_same_v<typename acc_t::offset_policy, acc_t>);
        static_assert(std::is_same_v<decltype(acc.access(ptr, i)), T&>);
        static_assert(std::is_same_v<decltype(acc.offset(ptr, i)),
                                     typename acc_t::data_handle_type>);
        static_assert(std::is_nothrow_move_constructible_v<acc_t>);
        static_assert(std::is_nothrow_move_assignable_v<acc_t>);
        static_assert(std::is_nothrow_swappable_v<acc_t>);
        static_assert(std::is_trivially_copyable_v<acc_t>);
        static_assert(std::is_trivially_default_constructible_v<acc_t>);
        static_assert(std::is_trivially_constructible_v<acc_t>);
        static_assert(std::is_trivially_move_constructible_v<acc_t>);
        static_assert(std::is_trivially_assignable_v<acc_t, acc_t>);
        static_assert(std::is_trivially_move_assignable_v<acc_t>);
        static_assert(std::copyable<acc_t>);
        static_assert(std::is_empty_v<acc_t>);
      },
      errors);
  ASSERT_EQ(errors, 0);
}

void test_restrict_accessor_conversion() {
  using ExecutionSpace = TEST_EXECSPACE;
  using T              = float;
  using memory_space   = typename ExecutionSpace::memory_space;
  using acc_t          = Kokkos::Impl::RestrictAccessor<T>;
  using const_acc_t    = Kokkos::Impl::RestrictAccessor<const T>;
  using int_acc_t      = Kokkos::Impl::RestrictAccessor<int>;
  using defacc_t       = Kokkos::default_accessor<T>;
  using const_defacc_t = Kokkos::default_accessor<const T>;
  using int_defacc_t   = Kokkos::default_accessor<int>;
  using rc_acc_t = Kokkos::Impl::ReferenceCountedAccessor<memory_space, acc_t>;
  using const_rc_acc_t =
      Kokkos::Impl::ReferenceCountedAccessor<memory_space, const_acc_t>;
  using rc_defacc_t =
      Kokkos::Impl::ReferenceCountedAccessor<memory_space, defacc_t>;
  using const_rc_defacc_t =
      Kokkos::Impl::ReferenceCountedAccessor<memory_space, const_defacc_t>;

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, 1), KOKKOS_LAMBDA(int) {
        static_assert(std::is_constructible_v<const_acc_t, acc_t>);
        static_assert(std::is_convertible_v<acc_t, const_acc_t>);
        static_assert(!std::is_constructible_v<acc_t, const_acc_t>);
        static_assert(!std::is_constructible_v<acc_t, int_acc_t>);
        static_assert(std::is_constructible_v<defacc_t, acc_t>);
        static_assert(std::is_constructible_v<acc_t, defacc_t>);
        static_assert(!std::is_constructible_v<int_defacc_t, acc_t>);
        static_assert(!std::is_constructible_v<int_acc_t, defacc_t>);
        static_assert(std::is_constructible_v<const_defacc_t, const_acc_t>);
        static_assert(std::is_constructible_v<const_acc_t, const_defacc_t>);
        static_assert(std::is_constructible_v<const_defacc_t, acc_t>);
        static_assert(std::is_constructible_v<const_acc_t, defacc_t>);
        static_assert(!std::is_constructible_v<defacc_t, const_acc_t>);
        static_assert(!std::is_constructible_v<acc_t, const_defacc_t>);
        static_assert(std::is_convertible_v<acc_t, const_acc_t>);
        static_assert(std::is_convertible_v<defacc_t, acc_t>);
        static_assert(std::is_convertible_v<defacc_t, const_acc_t>);
        static_assert(std::is_convertible_v<const_defacc_t, const_acc_t>);
        static_assert(!std::is_convertible_v<acc_t, defacc_t>);
        static_assert(!std::is_convertible_v<acc_t, const_defacc_t>);
        static_assert(!std::is_convertible_v<const_acc_t, const_defacc_t>);

        static_assert(std::is_constructible_v<const_rc_acc_t, rc_acc_t>);
        static_assert(!std::is_constructible_v<rc_acc_t, const_rc_acc_t>);
        static_assert(std::is_constructible_v<rc_acc_t, rc_defacc_t>);
        static_assert(std::is_constructible_v<rc_defacc_t, rc_acc_t>);
        static_assert(std::is_constructible_v<const_rc_acc_t, rc_defacc_t>);
        static_assert(std::is_constructible_v<const_rc_defacc_t, rc_acc_t>);
        static_assert(!std::is_constructible_v<rc_acc_t, const_rc_defacc_t>);
        static_assert(!std::is_constructible_v<rc_defacc_t, const_rc_acc_t>);
      });
}

TEST(TEST_CATEGORY, mdspan_restrict_accessor) {
  using ExecutionSpace = TEST_EXECSPACE;
  test_restrict_accessor<int, ExecutionSpace>();
  test_restrict_accessor<const int, ExecutionSpace>();
  test_restrict_accessor<double, ExecutionSpace>();
}

TEST(TEST_CATEGORY, mdspan_restrict_accessor_conversion) {
  test_restrict_accessor_conversion();
}
