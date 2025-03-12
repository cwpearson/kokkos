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

#include <Kokkos_Core.hpp>
#include "Benchmark_Context.hpp"
#include "PerfTest_Category.hpp"
#include <benchmark/benchmark.h>

namespace Test {

// FIXME: const input
template <typename Layout, typename Space, typename Scalar>
void stencil_basic(const Kokkos::View<Scalar**, Layout, Space>& input,
                   const Kokkos::View<Scalar**, Layout, Space>& output) {
  using range_policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;

  const int N = input.extent(0);
  const int M = input.extent(1);

  Kokkos::parallel_for(
      "stencil_basic", range_policy({1, 1}, {N - 1, M - 1}),
      KOKKOS_LAMBDA(const int i, const int j) {
        output(i, j) = (input(i - 1, j) + input(i + 1, j) + input(i, j - 1) +
                        input(i, j + 1)) /
                       (4 * input(i, j));
      });
}

// FIXME: const input
template <typename Layout, typename Space, typename Scalar>
void stencil_team(const Kokkos::View<Scalar**, Layout, Space>& input,
                  const Kokkos::View<Scalar**, Layout, Space>& output) {
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = typename team_policy::member_type;

  const int N = input.extent(0);
  const int M = input.extent(1);

  // Define tile dimensions
  const int tile_size_i = 32;  // Adjust tile size as needed
  const int tile_size_j = 32;  // Adjust tile size as needed

  const int nBlocks_i = (N - 2 + tile_size_i - 1) / tile_size_i;
  const int nBlocks_j = (M - 2 + tile_size_j - 1) / tile_size_j;

  Kokkos::parallel_for(
      "stencil_team",
      team_policy(nBlocks_i * nBlocks_j, Kokkos::AUTO, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type& team) {
        const int tile_i = team.league_rank() % (nBlocks_i);
        const int tile_j = team.league_rank() / (nBlocks_i);

        const int start_i = tile_i * tile_size_i + 1;
        const int start_j = tile_j * tile_size_j + 1;
        const int end_i   = Kokkos::min(start_i + tile_size_i, N - 1);
        const int end_j   = Kokkos::min(start_j + tile_size_j, M - 1);

        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, start_i, end_i), [&](const int i) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team, start_j, end_j),
                  [&](const int j) {
                    output(i, j) = (input(i - 1, j) + input(i + 1, j) +
                                    input(i, j - 1) + input(i, j + 1)) /
                                   (4 * input(i, j));
                  });
            });
      });
}

// FIXME: const input
template <typename Layout, typename Space, typename Scalar>
void stencil_team_scratch(const Kokkos::View<Scalar**, Layout, Space>& input,
                          const Kokkos::View<Scalar**, Layout, Space>& output) {
  using team_policy = Kokkos::TeamPolicy<>;
  using member_type = typename team_policy::member_type;

  const int N = input.extent(0);
  const int M = input.extent(1);

  // Define tile dimensions
  const int tile_size_i = 32;  // Adjust tile size as needed
  const int tile_size_j = 32;  // Adjust tile size as needed

  const int nBlocks_i = (N - 2 + tile_size_i - 1) / tile_size_i;
  const int nBlocks_j = (M - 2 + tile_size_j - 1) / tile_size_j;

  // Define the amount of scratch memory needed per team
  const int scratch_level = 0;
  const size_t bytes_per_team =
      (tile_size_i + 2) * (tile_size_j + 2) * sizeof(Scalar);

  Kokkos::parallel_for(
      "stencil_team_scratch",
      team_policy(nBlocks_i * nBlocks_j, Kokkos::AUTO, Kokkos::AUTO)
          .set_scratch_size(scratch_level, Kokkos::PerTeam(bytes_per_team)),
      KOKKOS_LAMBDA(const member_type& team) {
        // Allocate scratch memory
        Scalar* scratch =
            (Scalar*)team.team_scratch(scratch_level).get_shmem(bytes_per_team);

        const int tile_i = team.league_rank() % (nBlocks_i);
        const int tile_j = team.league_rank() / (nBlocks_i);

        const int start_i = tile_i * tile_size_i;
        const int start_j = tile_j * tile_size_j;
        const int end_i   = Kokkos::min(start_i + tile_size_i + 2, N);
        const int end_j   = Kokkos::min(start_j + tile_size_j + 2, M);

        // Load input tile into scratch memory
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, start_i, end_i), [&](const int i) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team, start_j, end_j),
                  [&](const int j) {
                    scratch[(i - start_i) * (tile_size_j + 2) + (j - start_j)] =
                        input(i, j);
                  });
            });

        team.team_barrier();

        // Perform stencil computation using scratch memory
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(team, 1, end_i - start_i - 1),
            [&](const int ti) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(team, 1, end_j - start_j - 1),
                  [&](const int tj) {
                    const int i = start_i + ti;
                    const int j = start_j + tj;
                    output(i, j) =
                        (scratch[(ti - 1) * (tile_size_j + 2) + tj] +
                         scratch[(ti + 1) * (tile_size_j + 2) + tj] +
                         scratch[ti * (tile_size_j + 2) + (tj - 1)] +
                         scratch[ti * (tile_size_j + 2) + (tj + 1)]) /
                        (4 * scratch[ti * (tile_size_j + 2) + tj]);
                  });
            });
      });
}

//--------------------------------------------------------------------------

template <typename Scalar, typename Layout, typename MemorySpace,
          typename StencilFn>
static double time_stencil(
    const Kokkos::View<Scalar**, Layout, MemorySpace>& input,
    const Kokkos::View<Scalar**, Layout, MemorySpace>& output, StencilFn&& fn) {
  Kokkos::Timer timer;
  fn(input, output);
  Kokkos::fence();
  return timer.seconds();
}

template <typename Scalar, typename StencilFn>
static void Stencil(benchmark::State& state, StencilFn&& stencil_fn) {
  using Layout      = Kokkos::LayoutRight;
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;

  const auto n = state.range(0);

  Kokkos::View<Scalar**, Layout, MemorySpace> input("input", n, n);
  Kokkos::View<Scalar**, Layout, MemorySpace> output("output", n, n);

  for (auto _ : state) {
    const auto time =
        time_stencil<double, Kokkos::LayoutRight,
                     typename Kokkos::DefaultExecutionSpace::memory_space>(
            input, output, stencil_fn);

    state.SetIterationTime(time);
    state.counters["time/input"] = benchmark::Counter(time / n / n);
    state.counters["bytes"] = benchmark::Counter(n * n * sizeof(Scalar) * 2);
  }
}

template <typename Scalar>
static void StencilBasic(benchmark::State& state) {
  using Layout      = Kokkos::LayoutRight;
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
  Stencil<Scalar>(state, stencil_basic<Layout, MemorySpace, Scalar>);
}

template <typename Scalar>
static void StencilTeam(benchmark::State& state) {
  using Layout      = Kokkos::LayoutRight;
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
  Stencil<Scalar>(state, stencil_team<Layout, MemorySpace, Scalar>);
}

template <typename Scalar>
static void StencilTeamScratch(benchmark::State& state) {
  using Layout      = Kokkos::LayoutRight;
  using MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space;
  Stencil<Scalar>(state, stencil_team_scratch<Layout, MemorySpace, Scalar>);
}

BENCHMARK(StencilBasic<double>)
    ->ArgName("n")
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->UseManualTime();

BENCHMARK(StencilTeam<double>)
    ->ArgName("n")
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->UseManualTime();

BENCHMARK(StencilTeamScratch<double>)
    ->ArgName("n")
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(20000)
    ->UseManualTime();

}  // namespace Test
