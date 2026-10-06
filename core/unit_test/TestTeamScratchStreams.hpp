// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

#include <gtest/gtest.h>

namespace Test {

namespace Impl {

// Adapter for the native stream/queue interop that backs independent execution
// space instances. Every specialization is guarded by its own enable macro so
// that several GPU backends can be enabled in the same build.
template <typename ExecSpace>
struct StreamInterOp;

#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct StreamInterOp<Kokkos::Cuda> {
  using exec_space = Kokkos::Cuda;
  using stream_t   = cudaStream_t;

  static stream_t create_stream() {
    stream_t stream;
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreate(&stream));
    return stream;
  }

  static exec_space exec_from_stream(stream_t stream) {
    return exec_space(stream);
  }

  static void release(exec_space& exec, stream_t& stream) {
    exec = exec_space();
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamDestroy(stream));
  }
};
#endif

#if defined(KOKKOS_ENABLE_HIP)
template <>
struct StreamInterOp<Kokkos::HIP> {
  using exec_space = Kokkos::HIP;
  using stream_t   = hipStream_t;

  static stream_t create_stream() {
    stream_t stream;
    KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamCreate(&stream));
    return stream;
  }

  static exec_space exec_from_stream(stream_t stream) {
    return exec_space(stream);
  }

  static void release(exec_space& exec, stream_t& stream) {
    exec = exec_space();
    KOKKOS_IMPL_HIP_SAFE_CALL(hipStreamDestroy(stream));
  }
};
#endif

#if defined(KOKKOS_ENABLE_SYCL)
template <>
struct StreamInterOp<Kokkos::SYCL> {
  using exec_space = Kokkos::SYCL;
  using stream_t   = sycl::queue;

  static stream_t create_stream() {
    // Queues have to share the context Kokkos was initialized with.
    sycl::context context = exec_space().sycl_queue().get_context();
    return sycl::queue(context, sycl::default_selector_v,
                       sycl::property::queue::in_order());
  }

  static exec_space exec_from_stream(stream_t stream) {
    return exec_space(stream);
  }

  static void release(exec_space&, stream_t&) {
    // Queues are reference counted and released together with the execution
    // space instances and the stream handles going out of scope.
  }
};
#endif

struct StreamScratchTestFunctor {
  using team_t = Kokkos::TeamPolicy<TEST_EXECSPACE>::member_type;
  using scratch_t =
      Kokkos::View<int64_t*, TEST_EXECSPACE::scratch_memory_space>;

  Kokkos::View<int64_t, TEST_EXECSPACE::memory_space,
               Kokkos::MemoryTraits<Kokkos::Atomic>>
      counter;
  int N, M;
  StreamScratchTestFunctor(
      Kokkos::View<int64_t, TEST_EXECSPACE::memory_space> counter_, int N_,
      int M_)
      : counter(counter_), N(N_), M(M_) {}

  KOKKOS_FUNCTION
  void operator()(const team_t& team) const {
    scratch_t scr(team.team_scratch(1), M);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, M),
                         [&](int i) { scr[i] = 0; });
    team.team_barrier();
    for (int i = 0; i < N; i++) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, M),
                           [&](int j) { scr[j] += 1; });
    }
    team.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0, M), [&](int i) {
      if (scr[i] != N) counter()++;
    });
  }
};

void stream_scratch_test_one(
    int N, int T, int M_base,
    Kokkos::View<int64_t, TEST_EXECSPACE::memory_space> counter,
    TEST_EXECSPACE exec, int tid) {
  int M = M_base + tid * 5;
  Kokkos::TeamPolicy<TEST_EXECSPACE> p(exec, T, 64);
  using scratch_t =
      Kokkos::View<int64_t*, TEST_EXECSPACE::scratch_memory_space>;

  int bytes = scratch_t::shmem_size(M);

  for (int r = 0; r < 15; r++) {
    Kokkos::parallel_for("Run", p.set_scratch_size(1, Kokkos::PerTeam(bytes)),
                         StreamScratchTestFunctor(counter, N, M));
  }
}

void stream_scratch_test(
    int N, int T, int M_base,
    Kokkos::View<int64_t, TEST_EXECSPACE::memory_space> counter) {
  using interop_t = StreamInterOp<TEST_EXECSPACE>;
  constexpr int K = 4;

  typename interop_t::stream_t stream[K];
  TEST_EXECSPACE exec[K];
  for (int i = 0; i < K; i++) {
    stream[i] = interop_t::create_stream();
    exec[i]   = interop_t::exec_from_stream(stream[i]);
  }

  // Test that growing scratch size in subsequent calls doesn't crash things
#if defined(KOKKOS_ENABLE_OPENMP)
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    // Limit how many threads submit
    if (tid < 4) {
      stream_scratch_test_one(N, T, M_base, counter, exec[tid], tid);
    }
  }
#else
  for (int tid = 0; tid < K; tid++) {
    stream_scratch_test_one(N, T, M_base, counter, exec[tid], tid);
  }
#endif
  // Test that if everything is large enough, multiple launches with different
  // scratch sizes don't step on each other
  for (int tid = K - 1; tid >= 0; tid--) {
    stream_scratch_test_one(N, T, M_base, counter, exec[tid], tid);
  }

  Kokkos::fence();
  for (int i = 0; i < K; i++) {
    interop_t::release(exec[i], stream[i]);
  }
}
}  // namespace Impl

TEST(TEST_CATEGORY, team_scratch_1_streams) {
  int N      = 10000;
  int T      = 10;
  int M_base = 150;

  Kokkos::View<int64_t, TEST_EXECSPACE::memory_space> counter("C");

  Impl::stream_scratch_test(N, T, M_base, counter);

  int64_t result;
  Kokkos::deep_copy(result, counter);
  ASSERT_EQ(0, result);
}
}  // namespace Test
