kokkos_core_unit_test_generate_backend_sources(Serial)

# FIXME_NEXTSILICON: requires parallel_for
if(Kokkos_ENABLE_IMPL_MDSPAN AND NOT KOKKOS_ENABLE_NEXTSILICON)
  kokkos_add_executable_and_test(CoreUnitTest_Serial_ViewSupport SOURCES UnitTestMainInit.cpp ${Serial_VIEWSUPPORT})
endif()

# Fails serial.atomics_tpetra_max_abs when we test with Clacc (same as CoreUnitTest_Serial1).
if(KOKKOS_ENABLE_OPENACC AND KOKKOS_CXX_COMPILER_ID STREQUAL Clang)

  list(REMOVE_ITEM Serial_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Atomics.cpp)
endif()
if(KOKKOS_ENABLE_NEXTSILICON)
  #FIXME_NEXTSILICON: requires parallel_for tagged dispatch
  list(REMOVE_ITEM Serial_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Atomics.cpp)
  #FIXME_NEXTSILICON: requires TeamPolicy parallel_for on NextSilicon
  list(REMOVE_ITEM Serial_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_DeepCopy_Assignment.cpp)
  list(REMOVE_ITEM Serial_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Graph.cpp)
  #FIXME_NEXTSILICON: requires TeamPolicy parallel_reduce on NextSilicon
  list(REMOVE_ITEM Serial_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_CommonPolicyInterface.cpp)
  list(REMOVE_ITEM Serial_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Reductions.cpp)
endif()
kokkos_add_executable_and_test(CoreUnitTest_Serial_SmokeTest SOURCES UnitTestMainInit.cpp ${Serial_SOURCES_SMOKE})

if(Kokkos_ENABLE_NEXTSILICON)
  #FIXME_NEXTSILICON: some Serial tests require operations on the default device
  list(
    REMOVE_ITEM
    Serial_SOURCES1
    #FIXME_NEXTSILICON: requires TeamPolicy<NextSilicon>
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_CommonPolicyConstructors.cpp
    #FIXME_NEXTSILICON: requires TeamPolicy parallel_for on NextSilicon
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_DeepCopy_Assignment.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_DeepCopy_Narrowing.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Graph.cpp
    #FIXME_NEXTSILICON: requires long double support
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_MathematicalConstants.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_MathematicalFunctions1.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_MathematicalFunctions2.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_MathematicalFunctions3.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_NumericTraits.cpp
    #FIXME_NEXTSILICON requires parallel_for
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_MinMaxClamp.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Other.cpp
    #FIXME_NEXTSILICON requires parallel_reduce
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_RangePolicyExecutionTypes.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_CommonPolicyInterface.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Reducers_a.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Reducers_b.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Reducers_c.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/serial/TestSerial_Reductions.cpp
  )
endif()

kokkos_add_executable_and_test(CoreUnitTest_Serial1 SOURCES UnitTestMainInit.cpp ${Serial_SOURCES1})
kokkos_add_executable_and_test(CoreUnitTest_Serial2 SOURCES UnitTestMainInit.cpp ${Serial_SOURCES2})
