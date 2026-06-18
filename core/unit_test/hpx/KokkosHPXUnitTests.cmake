kokkos_core_unit_test_generate_backend_sources(HPX)

# FIXME_NEXTSILICON: requires parallel_for
if(Kokkos_ENABLE_IMPL_MDSPAN AND NOT KOKKOS_ENABLE_NEXTSILICON)
  kokkos_add_executable_and_test(CoreUnitTest_HPX_ViewSupport SOURCES UnitTestMainInit.cpp ${HPX_VIEWSUPPORT})
endif()
kokkos_add_executable_and_test(CoreUnitTest_HPX_SmokeTest SOURCES UnitTestMainInit.cpp ${HPX_SOURCES_SMOKE})
kokkos_add_executable_and_test(CoreUnitTest_HPX SOURCES UnitTestMainInit.cpp ${HPX_SOURCES})
kokkos_add_executable_and_test(CoreUnitTest_HPXInterOp SOURCES UnitTestMain.cpp hpx/TestHPX_InterOp.cpp)
kokkos_add_executable_and_test(
  CoreUnitTest_HPX_IndependentInstances
  SOURCES
  UnitTestMainInit.cpp
  hpx/TestHPX_IndependentInstances.cpp
  hpx/TestHPX_IndependentInstancesDelayedExecution.cpp
  hpx/TestHPX_IndependentInstancesInstanceIds.cpp
  hpx/TestHPX_IndependentInstancesRefCounting.cpp
  hpx/TestHPX_IndependentInstancesSynchronization.cpp
)
