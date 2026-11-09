// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Compares an otherwise-identical kernel compiled with and without
// Kokkos::Restrict so the generated code can be diffed.
//
// The kernel body is
//
//   out(i)  = in(i);
//   out(i) += in(i);
//
// Without restrict, the store to out(i) may alias in(i), so the compiler must
// reload in(i) for the second statement and emit two stores to out(i):
//
//   load in[i]; store out[i]; load in[i]; load out[i]; fadd; store out[i]
//
// With restrict honored, the reload and the first store are dead and the body
// should collapse to roughly:
//
//   load in[i]; fadd (or fmul 2.0); store out[i]
//
// Both variants use Unmanaged so that the only difference between the two
// View types is the Restrict trait (Restrict without Unmanaged wraps the
// RestrictAccessor in a ReferenceCountedAccessor, which changes the data
// handle type).

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>

namespace {

using NormalTraits = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
using RestrictTraits =
    Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Restrict>;

template <class Traits>
struct DoubleKernel {
  Kokkos::View<double*, Traits> out;
  Kokkos::View<const double*, Traits> in;

  KOKKOS_FUNCTION void operator()(int i) const {
    out(i) = in(i);
    out(i) += in(i);
  }
};

// Named, non-inlined launchers give each variant an easy-to-find symbol.
template <class Traits>
[[gnu::noinline]] void run(const char* label, const Kokkos::View<double*>& out,
                           const Kokkos::View<double*>& in) {
  DoubleKernel<Traits> k{out, in};
  Kokkos::parallel_for(label, out.extent(0), k);
  Kokkos::fence();
}

bool check(const char* label, const Kokkos::View<double*>& out) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
  for (size_t i = 0; i < h.extent(0); ++i) {
    if (h(i) != 2.0 * static_cast<double>(i)) {
      std::printf("[%s] out(%zu) = %g, expected %g  FAIL\n", label, i, h(i),
                  2.0 * static_cast<double>(i));
      return false;
    }
  }
  std::printf("[%s] PASS\n", label);
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  bool ok = true;
  {
    // Runtime size so the loop can't be fully constant-folded.
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;

    Kokkos::View<double*> in("in", n), out_normal("out_normal", n),
        out_restrict("out_restrict", n);
    Kokkos::parallel_for(
        "init", n, KOKKOS_LAMBDA(int i) { in(i) = static_cast<double>(i); });
    Kokkos::fence();

    run<NormalTraits>("normal", out_normal, in);
    run<RestrictTraits>("restrict", out_restrict, in);

    ok = check("normal", out_normal) && ok;
    ok = check("restrict", out_restrict) && ok;
  }
  Kokkos::finalize();
  return ok ? 0 : 1;
}
