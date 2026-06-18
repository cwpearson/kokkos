kokkos_core_unit_test_generate_backend_sources(SYCL HOST_ACCESSIBLE_CATEGORY SYCLSharedUSM)
kokkos_core_unit_test_generate_memory_space_sources(SYCL HostUSM)
kokkos_core_unit_test_generate_memory_space_sources(SYCL SharedUSM)

if(Kokkos_ENABLE_IMPL_MDSPAN)
  kokkos_add_executable_and_test(CoreUnitTest_SYCL_ViewSupport SOURCES UnitTestMainInit.cpp ${SYCL_VIEWSUPPORT})
endif()

# WorkGraph is not supported for SYCL in unit tests (historically excluded from CoreUnitTest_SYCL2A).
list(REMOVE_ITEM SYCL_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/sycl/TestSYCL_WorkGraph.cpp)
kokkos_add_executable_and_test(CoreUnitTest_SYCL_SmokeTest SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES_SMOKE})

kokkos_add_executable_and_test(CoreUnitTest_SYCL1A SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES1A})

kokkos_add_executable_and_test(CoreUnitTest_SYCL1B SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES1B})

kokkos_add_executable_and_test(CoreUnitTest_SYCL2A SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES2A})

kokkos_add_executable_and_test(CoreUnitTest_SYCL2B SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES2B})

kokkos_add_executable_and_test(CoreUnitTest_SYCL2C SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES2C})

kokkos_add_executable_and_test(CoreUnitTest_SYCL2D SOURCES UnitTestMainInit.cpp ${SYCL_SOURCES2D})

kokkos_add_executable_and_test(
  CoreUnitTest_SYCL3 SOURCES UnitTestMainInit.cpp sycl/TestSYCL_TeamScratchStreams.cpp ${SYCL_SOURCES3}
  sycl/TestSYCL_Spaces.cpp
)

kokkos_add_executable_and_test(CoreUnitTest_SYCLInterOpInit SOURCES UnitTestMain.cpp sycl/TestSYCL_InterOp_Init.cpp)
kokkos_add_executable_and_test(
  CoreUnitTest_SYCLInterOpInit_Context SOURCES UnitTestMainInit.cpp sycl/TestSYCL_InterOp_Init_Context.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_SYCLInterOpStreams SOURCES UnitTestMain.cpp sycl/TestSYCL_InterOp_Streams.cpp
)
kokkos_add_executable_and_test(
  CoreUnitTest_SYCLInterOpStreamsMultiGPU SOURCES UnitTestMainInit.cpp sycl/TestSYCL_InterOp_StreamsMultiGPU.cpp
)

if(KOKKOS_IMPL_HAVE_SYCL_EXT_ONEAPI_GRAPH AND NOT Kokkos_ENABLE_IMPL_SYCL_OUT_OF_ORDER_QUEUES)
  kokkos_add_executable_and_test(
    CoreUnitTest_SYCLInterOpGraph SOURCES UnitTestMainInit.cpp sycl/TestSYCL_InterOp_Graph.cpp
  )
endif()
