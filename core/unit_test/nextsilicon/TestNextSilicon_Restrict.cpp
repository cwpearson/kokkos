// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double> v("backing");
    Kokkos::View<double, Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Unmanaged>> urv(v);

    Kokkos::parallel_for(
        "write", 1, KOKKOS_LAMBDA(int) { urv() = 42.0; });

    std::printf("[Scalar View] value = %g  %s\n", v(),
                v() == 42.0 ? "PASS" : "FAIL");
  }
  Kokkos::finalize();
  return 0;
}
