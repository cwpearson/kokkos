// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// @Kokkos_Feature_Level_Required:11
// Unit test for parallel_reduce with TeamPolicy

#include <gtest/gtest.h>
#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace Test {

template <class ExecSpace>
struct ParallelReduceTeamPolicy {
  using policy_t   = Kokkos::TeamPolicy<ExecSpace>;
  using team_t     = typename policy_t::member_type;
  using value_type = long long;

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_t &team, value_type &update) const {
    update += team.league_rank() * team.team_size() + team.team_rank();
  }

  void run(const int league_size, int team_size) const {
    const int max_team_size =
        policy_t(1, Kokkos::AUTO)
            .team_size_max(*this, Kokkos::ParallelReduceTag{});
    if (team_size > max_team_size) team_size = max_team_size;

    policy_t policy(league_size, team_size);

    ASSERT_EQ(policy.league_size(), league_size);
    ASSERT_LE(policy.team_size(), team_size);

    value_type result = 0;
    Kokkos::parallel_reduce("Teams", policy, *this, result);

    const value_type work_size = value_type(league_size) * policy.team_size();
    ASSERT_EQ(result, work_size * (work_size - 1) / 2);
  }
};

TEST(TEST_CATEGORY, IncrTest_11_ParallelReduce_TeamPolicy) {
  ParallelReduceTeamPolicy<TEST_EXECSPACE> test;

  test.run(1, 4);
  test.run(8, 16);
  test.run(11, 13);
}

}  // namespace Test
