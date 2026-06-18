kokkos_core_unit_test_generate_backend_sources(Cuda HOST_ACCESSIBLE_CATEGORY CudaUVM)
kokkos_core_unit_test_generate_memory_space_sources(Cuda HostPinned)
kokkos_core_unit_test_generate_memory_space_sources(Cuda UVM)

if(Kokkos_ENABLE_IMPL_MDSPAN)
  kokkos_add_executable_and_test(CoreUnitTest_Cuda_ViewSupport SOURCES UnitTestMainInit.cpp ${Cuda_VIEWSUPPORT})
endif()
kokkos_add_executable_and_test(CoreUnitTest_Cuda_SmokeTest SOURCES UnitTestMainInit.cpp ${Cuda_SOURCES_SMOKE})

kokkos_add_executable_and_test(
  CoreUnitTest_Cuda1 SOURCES UnitTestMainInit.cpp ${Cuda_SOURCES1} cuda/TestCuda_ReducerViewSizeLimit.cpp
)

kokkos_add_executable_and_test(CoreUnitTest_Cuda2 SOURCES UnitTestMainInit.cpp ${Cuda_SOURCES2})

kokkos_add_executable_and_test(
  CoreUnitTest_Cuda3
  SOURCES
  UnitTestMainInit.cpp
  cuda/TestCuda_TeamScratchStreams.cpp
  ${Cuda_SOURCES3}
  cuda/TestCuda_Spaces.cpp
  ${Cuda_SOURCES_SHAREDSPACE}
)

# This test seeks to make sure that `desul::ensure_cuda_lock_arrays_on_device` is called before a graph submission.
# It is in a separate file to ensure that the test takes place in a separate compilation unit, in which this function
# has not been called yet by Kokkos::initialize or by a preceding kernel launch.
set(file ${CMAKE_CURRENT_BINARY_DIR}/cuda/TestCuda_GraphAtomicLocks.cpp)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cuda/dummy.cpp "#include <TestCuda_Category.hpp>\n"
                                                      "#include <TestGraphAtomicLocks.hpp>\n"
)
configure_file(${CMAKE_CURRENT_BINARY_DIR}/cuda/dummy.cpp ${file})
kokkos_add_executable_and_test(CoreUnitTest_CudaGraphAtomicLocks SOURCES UnitTestMainInit.cpp ${file})

kokkos_add_executable_and_test(
  CoreUnitTest_CudaGraphScratch SOURCES UnitTestMainInit.cpp cuda/TestCuda_GraphScratch.cpp
)

kokkos_add_executable_and_test(
  CoreUnitTest_CudaLargeScratch SOURCES UnitTestMainInit.cpp cuda/TestCuda_LargeScratch.cpp
)

kokkos_add_executable_and_test(
  CoreUnitTest_CudaTimingBased SOURCES UnitTestMainInit.cpp cuda/TestCuda_DebugSerialExecution.cpp
  cuda/TestCuda_DebugPinUVMSpace.cpp
)

kokkos_add_executable_and_test(CoreUnitTest_CudaInterOpInit SOURCES UnitTestMain.cpp cuda/TestCuda_InterOp_Init.cpp)
kokkos_add_executable_and_test(
  CoreUnitTest_CudaInterOpStreams SOURCES UnitTestMain.cpp cuda/TestCuda_InterOp_Streams.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_CudaInterOpStreamsMultiGPU SOURCES UnitTestMainInit.cpp cuda/TestCuda_InterOp_StreamsMultiGPU.cpp
)

kokkos_add_executable_and_test(
  CoreUnitTest_CudaInterOpGraph SOURCES UnitTestMainInit.cpp cuda/TestCuda_InterOp_Graph.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_CudaInterOpGraphMultiGPU SOURCES UnitTestMainInit.cpp cuda/TestCuda_InterOp_GraphMultiGPU.cpp
)
