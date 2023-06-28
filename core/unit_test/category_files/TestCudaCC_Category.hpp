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

#ifndef KOKKOS_TEST_CUDACC_HPP
#define KOKKOS_TEST_CUDACC_HPP

#include <gtest/gtest.h>

#define TEST_CATEGORY cuda_cc
#define TEST_CATEGORY_DEATH cuda_cc_DeathTest
#define TEST_EXECSPACE Kokkos::CudaCCSpace

#endif
