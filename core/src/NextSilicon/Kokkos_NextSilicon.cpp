// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_InitializationCallbacks.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>
#include <impl/Kokkos_CheckUsage.hpp>

#include <nextapi/memory.h>
#include <nextapi/parallelism.hpp>

#include <cstdlib>
#include <new>
#include <ostream>

namespace {

constexpr std::size_t nextsilicon_debug_page_size = 1 << 20;

template <class T>
T* nextsilicon_debug_host_pinned_new() {
  void* storage = std::aligned_alloc(nextsilicon_debug_page_size,
                                     nextsilicon_debug_page_size);
  if (!storage) throw std::bad_alloc();

  try {
    auto* ptr = new (storage) T;
    nextapi_mem_migrate(storage, nextsilicon_debug_page_size,
                        NEXTAPI_PAGE_LOC_HOST, /*pin=*/true);
    return ptr;
  } catch (...) {
    std::free(storage);
    throw;
  }
}

template <class T>
void nextsilicon_debug_host_pinned_delete(T* ptr) {
  if (!ptr) return;
  ptr->~T();
  std::free(ptr);
}

}  // namespace

Kokkos::Experimental::NextSilicon::NextSilicon()
    : m_space_instance(
          (Kokkos::Impl::check_execution_space_constructor_precondition(name()),
           Impl::NextSiliconInternal::default_instance)) {}

Kokkos::Experimental::NextSilicon::~NextSilicon() {
  Kokkos::Impl::check_execution_space_destructor_precondition(name());
}

void Kokkos::Experimental::NextSilicon::impl_initialize(
    InitializationSettings const& /*settings*/) {
  Kokkos::Impl::run_nextsilicon_initialization_callbacks();
  auto* internal_instance =
      nextsilicon_debug_host_pinned_new<Impl::NextSiliconInternal>();
  Impl::NextSiliconInternal::default_instance =
      Kokkos::Impl::HostSharedPtr<Impl::NextSiliconInternal>(
          internal_instance,
          nextsilicon_debug_host_pinned_delete<Impl::NextSiliconInternal>);
}

void Kokkos::Experimental::NextSilicon::impl_finalize() {
  Impl::NextSiliconInternal::default_instance = nullptr;
}

void Kokkos::Experimental::NextSilicon::print_configuration(
    std::ostream& os, bool /* verbose */) const {
  os << "Device Execuction Space:\n";
  os << "  KOKKOS_ENABLE_NEXTSILICON: yes\n";

  os << "\nNextSilicon Runtime Configuration:\n";
  m_space_instance->print_configuration(os);
}

void Kokkos::Experimental::NextSilicon::fence(std::string const& name) const {
  m_space_instance->fence(name);
}

void Kokkos::Experimental::NextSilicon::impl_static_fence(
    std::string const& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::NextSilicon>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [&]() { /*FIXME_NEXTSILICON*/ });
}

int Kokkos::Experimental::NextSilicon::impl_hardware_thread_id() noexcept {
  return nextapi::get_thread_index();
}

uint32_t Kokkos::Experimental::NextSilicon::impl_instance_id() const noexcept {
  return m_space_instance->instance_id();
}

namespace Kokkos {
namespace Impl {

// 180 is after OpenACC (170)
int g_nextsilicon_space_factory_initialized =
    initialize_space_factory<Experimental::NextSilicon>("180_NextSilicon");
}  // namespace Impl
}  // Namespace Kokkos
