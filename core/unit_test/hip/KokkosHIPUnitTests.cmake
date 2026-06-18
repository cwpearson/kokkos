kokkos_core_unit_test_generate_backend_sources(HIP HOST_ACCESSIBLE_CATEGORY HIPManaged)
kokkos_core_unit_test_generate_memory_space_sources(HIP HostPinned)
kokkos_core_unit_test_generate_memory_space_sources(HIP Managed)

if(Kokkos_ENABLE_IMPL_MDSPAN)
  kokkos_add_executable_and_test(CoreUnitTest_HIP_ViewSupport SOURCES UnitTestMainInit.cpp ${HIP_VIEWSUPPORT})
endif()
kokkos_add_executable_and_test(CoreUnitTest_HIP_SmokeTest SOURCES UnitTestMainInit.cpp ${HIP_SOURCES_SMOKE})

kokkos_add_executable_and_test(
  CoreUnitTest_HIP
  SOURCES
  UnitTestMainInit.cpp
  ${HIP_SOURCES}
  hip/TestHIP_ScanUnit.cpp
  hip/TestHIP_SharedResourceLock.cpp
  hip/TestHIP_Spaces.cpp
  hip/TestHIP_Memory_Requirements.cpp
  hip/TestHIP_TeamScratchStreams.cpp
  hip/TestHIP_AsyncLauncher.cpp
  hip/TestHIP_BlocksizeDeduction.cpp
  hip/TestHIP_UnifiedMemory_ZeroMemset.cpp
)

set(file ${CMAKE_CURRENT_BINARY_DIR}/hip/TestHIP_GraphAtomicLocks.cpp)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/hip/dummy.cpp "#include <TestHIP_Category.hpp>\n"
                                                     "#include <TestGraphAtomicLocks.hpp>\n"
)
configure_file(${CMAKE_CURRENT_BINARY_DIR}/hip/dummy.cpp ${file})
kokkos_add_executable_and_test(CoreUnitTest_HIPGraphAtomicLocks SOURCES UnitTestMainInit.cpp ${file})

kokkos_add_executable_and_test(CoreUnitTest_HIPInterOpInit SOURCES UnitTestMain.cpp hip/TestHIP_InterOp_Init.cpp)
kokkos_add_executable_and_test(
  CoreUnitTest_HIPInterOpStreams SOURCES UnitTestMain.cpp hip/TestHIP_InterOp_Streams.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_HIPInterOpGraph SOURCES UnitTestMainInit.cpp hip/TestHIP_InterOp_Graph.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_HIPInterOpStreamsMultiGPU SOURCES UnitTestMainInit.cpp hip/TestHIP_InterOp_StreamsMultiGPU.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_HIPInterOpGraphMultiGPU SOURCES UnitTestMainInit.cpp hip/TestHIP_InterOp_GraphMultiGPU.cpp
)
