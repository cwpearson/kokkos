// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

namespace {
using element_t      = float;
using memory_space_t = TEST_EXECSPACE::memory_space;
using defacc_t       = Kokkos::default_accessor<element_t>;
using const_defacc_t = Kokkos::default_accessor<const element_t>;
using acc_t = Kokkos::Impl::ReferenceCountedAccessor<memory_space_t, defacc_t>;
using const_acc_t =
    Kokkos::Impl::ReferenceCountedAccessor<memory_space_t, const_defacc_t>;
using data_handle_t       = typename acc_t::data_handle_type;
using const_data_handle_t = typename const_acc_t::data_handle_type;

template <class ElementType>
struct TestDataHandle {
  ElementType* ptr = nullptr;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestDataHandle() = default;

  KOKKOS_FUNCTION
  constexpr TestDataHandle(ElementType* arg_ptr) : ptr(arg_ptr) {}

  template <
      class OtherElementType,
      std::enable_if_t<std::is_convertible_v<OtherElementType*, ElementType*>,
                       int> = 0>
  KOKKOS_FUNCTION constexpr TestDataHandle(
      TestDataHandle<OtherElementType> other)
      : ptr(other.ptr) {}
};

template <class ElementType>
KOKKOS_INLINE_FUNCTION constexpr ElementType* ptr_from_data_handle(
    const TestDataHandle<ElementType>& handle) {
  return handle.ptr;
}

template <class ElementType>
struct TestOffsetDataHandle {
  ElementType* ptr = nullptr;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestOffsetDataHandle() = default;

  KOKKOS_FUNCTION
  constexpr TestOffsetDataHandle(ElementType* arg_ptr) : ptr(arg_ptr) {}

  template <
      class OtherElementType,
      std::enable_if_t<std::is_convertible_v<OtherElementType*, ElementType*>,
                       int> = 0>
  KOKKOS_FUNCTION constexpr TestOffsetDataHandle(
      TestOffsetDataHandle<OtherElementType> other)
      : ptr(other.ptr) {}
};

template <class ElementType>
struct TestAccessor;

template <class ElementType>
struct TestOffsetAccessor {
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = TestOffsetDataHandle<ElementType>;
  using offset_policy    = TestOffsetAccessor;

  int state = std::is_const_v<ElementType> ? -1 : 42;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestOffsetAccessor() = default;

  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   ElementType (*)[]>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr TestOffsetAccessor(
      const TestOffsetAccessor<OtherElementType>& other)
      : state(other.state) {}

  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   ElementType (*)[]>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr TestOffsetAccessor(
      const TestAccessor<OtherElementType>& other)
      : state(other.state) {}

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p.ptr[i];
  }

  KOKKOS_FUNCTION
  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return data_handle_type{p.ptr + i};
  }
};

template <class ElementType>
struct TestAccessor {
  using element_type     = ElementType;
  using reference        = ElementType&;
  using data_handle_type = TestDataHandle<ElementType>;
  using offset_policy    = TestOffsetAccessor<ElementType>;

  int state = std::is_const_v<ElementType> ? -1 : 42;

  KOKKOS_DEFAULTED_FUNCTION
  constexpr TestAccessor() = default;

  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<OtherElementType (*)[],
                                                   ElementType (*)[]>,
                             int> = 0>
  KOKKOS_FUNCTION constexpr TestAccessor(
      const TestAccessor<OtherElementType>& other)
      : state(other.state) {}

  KOKKOS_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p.ptr[i];
  }

  KOKKOS_FUNCTION
  constexpr typename offset_policy::data_handle_type offset(
      data_handle_type p, size_t i) const noexcept {
    return typename offset_policy::data_handle_type{p.ptr + i};
  }
};

using nested_acc_t =
    Kokkos::Impl::ReferenceCountedAccessor<memory_space_t,
                                           TestAccessor<element_t>>;
using const_nested_acc_t =
    Kokkos::Impl::ReferenceCountedAccessor<memory_space_t,
                                           TestAccessor<const element_t>>;
}  // namespace

TEST(TEST_CATEGORY, RefCountedAcc_Typedefs) {
  static_assert(std::is_same_v<typename acc_t::element_type, element_t>);
  static_assert(
      std::is_same_v<
          typename acc_t::data_handle_type,
          Kokkos::Impl::ReferenceCountedDataHandle<element_t, memory_space_t>>);
  static_assert(
      std::is_same_v<typename acc_t::reference, typename defacc_t::reference>);
  static_assert(
      std::is_same_v<typename acc_t::offset_policy,
                     Kokkos::Impl::ReferenceCountedAccessor<
                         memory_space_t, typename defacc_t::offset_policy>>);
}

TEST(TEST_CATEGORY, RefCountedAcc_NestedHandleTypedefs) {
  using expected_data_handle_t =
      Kokkos::Impl::ReferenceCountedDataHandle<element_t, memory_space_t,
                                               TestDataHandle<element_t>>;
  using expected_offset_handle_t =
      Kokkos::Impl::ReferenceCountedDataHandle<element_t, memory_space_t,
                                               TestOffsetDataHandle<element_t>>;

  static_assert(std::is_same_v<typename nested_acc_t::data_handle_type,
                               expected_data_handle_t>);
  static_assert(
      std::is_same_v<typename nested_acc_t::offset_policy::data_handle_type,
                     expected_offset_handle_t>);
  static_assert(
      std::is_same_v<
          decltype(std::declval<nested_acc_t>().offset(
              std::declval<const typename nested_acc_t::data_handle_type&>(),
              size_t{})),
          expected_offset_handle_t>);
}

TEST(TEST_CATEGORY, RefCountedAcc_NestedHandleAccessAndOffset) {
  auto shared_alloc =
      Kokkos::Impl::make_shared_allocation_record<element_t, memory_space_t,
                                                  TEST_EXECSPACE>(
          100, "Test", memory_space_t(),
          std::optional<TEST_EXECSPACE>(std::nullopt),
          std::bool_constant<true>(),    // init
          std::bool_constant<false>());  // sequential_host_init

  element_t* ptr = static_cast<element_t*>(shared_alloc->data());
  typename nested_acc_t::data_handle_type handle(shared_alloc);
  ASSERT_EQ(handle.use_count(), 1);
  ASSERT_EQ(handle.get().ptr, ptr);
  element_t* raw_ptr(handle);
  ASSERT_EQ(raw_ptr, ptr);

  nested_acc_t accessor;
  auto offset_handle = accessor.offset(handle, 5);
  EXPECT_EQ(offset_handle.get().ptr, ptr + 5);
  EXPECT_EQ(offset_handle.use_count(), 2);
  EXPECT_EQ(offset_handle.get_record(), handle.get_record());

  Kokkos::View<int, TEST_EXECSPACE> errors("Errors");
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        if (&accessor.access(handle, 5) != ptr + 5) errors() += 1;
      });
  int h_errors = 0;
  Kokkos::deep_copy(h_errors, errors);
  EXPECT_EQ(h_errors, 0);
}

TEST(TEST_CATEGORY, RefCountedAcc_PreservesNestedAccessorState) {
  nested_acc_t accessor;
  const_nested_acc_t const_accessor(accessor);

  EXPECT_EQ(accessor.nested_accessor().state, 42);
  EXPECT_EQ(const_accessor.nested_accessor().state, 42);

  typename nested_acc_t::offset_policy offset_accessor(accessor);
  EXPECT_EQ(offset_accessor.nested_accessor().state, 42);
}

template <class T>
KOKKOS_FUNCTION void unused_variable_sink(T) {}

void test_refcountedacc_ctors() {
  Kokkos::parallel_for(Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
      // default ctor and non-const to const
      {
        acc_t acc;
        const_acc_t c_acc(acc);
	static_assert(!std::is_constructible_v<acc_t, const_acc_t>);

	unused_variable_sink(c_acc);
}
// from default_accessor
{
  defacc_t defacc;
  const_defacc_t c_defacc;
  acc_t acc(defacc);
  const_acc_t c_acc1(defacc);
  const_acc_t c_acc2(c_defacc);
  static_assert(!std::is_constructible_v<acc_t, const_defacc_t>);

  unused_variable_sink(acc);
  unused_variable_sink(c_acc1);
  unused_variable_sink(c_acc2);
}
});
}

TEST(TEST_CATEGORY, RefCountedAcc_Ctors) { test_refcountedacc_ctors(); }

void test_refcountedacc_conversion_to_default_acc() {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        // default ctor and non-const to const
        acc_t acc;
        const_acc_t c_acc;
        defacc_t defacc(acc);
        const_defacc_t c_defacc1(acc);
        const_defacc_t c_defacc2(c_acc);
        (void)defacc;
        (void)c_defacc1;
        (void)c_defacc2;
        static_assert(!std::is_constructible_v<defacc_t, const_acc_t>);
      });
}

TEST(TEST_CATEGORY, RefCountedAcc_ConversionToDefaultAcc) {
  test_refcountedacc_conversion_to_default_acc();
}

void test_refcountedacc_access() {
  element_t* ptr = static_cast<element_t*>(
      Kokkos::kokkos_malloc<TEST_EXECSPACE::memory_space>(100 *
                                                          sizeof(element_t)));
  // Gonna use unmanaged data handles here (i.e. not actually referfence
  // counted)
  data_handle_t dh(ptr);
  const_data_handle_t cdh(ptr);

  Kokkos::View<int, TEST_EXECSPACE> errors("Errors");
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        acc_t acc;
        const_acc_t c_acc;
        if (&acc.access(dh, 5) != ptr + 5) errors() += 1;
        if (&c_acc.access(cdh, 5) != ptr + 5) errors() += 2;
      });
  int h_errors = 0;
  Kokkos::deep_copy(h_errors, errors);
  ASSERT_FALSE(h_errors & 1);
  ASSERT_FALSE(h_errors & 2);
  Kokkos::kokkos_free<TEST_EXECSPACE>(ptr);
}

TEST(TEST_CATEGORY, RefCountedAcc_Access) { test_refcountedacc_access(); }

void test_refcountedacc_conversion() {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<TEST_EXECSPACE>(0, 1), KOKKOS_LAMBDA(int) {
        using acc_anonym_t =
            Kokkos::Impl::ReferenceCountedAccessor<Kokkos::AnonymousSpace,
                                                   defacc_t>;
        using const_acc_anonym_t =
            Kokkos::Impl::ReferenceCountedAccessor<Kokkos::AnonymousSpace,
                                                   const_defacc_t>;
        acc_t acc;
        const_acc_t c_acc(acc);
        acc_anonym_t acc_anonym(acc);
        const_acc_anonym_t c_acc_anonym(acc);
        acc   = acc_anonym;
        c_acc = acc_anonym;
        static_assert(!std::is_constructible_v<acc_t, const_acc_t>);
        static_assert(!std::is_constructible_v<acc_anonym_t, const_acc_t>);
        static_assert(
            !std::is_constructible_v<acc_anonym_t, const_acc_anonym_t>);
        static_assert(!std::is_constructible_v<
                      Kokkos::Impl::ReferenceCountedAccessor<
                          memory_space_t, Kokkos::default_accessor<double>>,
                      acc_t>);

        unused_variable_sink(c_acc);
        unused_variable_sink(c_acc_anonym);
      });
}

TEST(TEST_CATEGORY, RefCountedAcc_Conversion) {
  test_refcountedacc_conversion();
}
