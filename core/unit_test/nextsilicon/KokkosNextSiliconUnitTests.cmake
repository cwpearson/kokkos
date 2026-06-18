kokkos_core_unit_test_generate_backend_sources(NextSilicon)

#FIXME_NEXTSILICON: requires parallel_for tagged dispatch
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_Atomics.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_BlockSizeDeduction.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_Complex.cpp)
#FIXME_NEXTSILICON: requires MDRange parallel_for on NextSilicon
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_ViewAPI_a.cpp)
#FIXME_NEXTSILICON: requires TeamPolicy parallel_for on NextSilicon
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_DeepCopy_Assignment.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_TeamBasic.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_TeamMDRange.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_TeamVectorRange.cpp)
#FIXME_NEXTSILICON: requires WorkGraphPolicy parallel_for on NextSilicon
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_WorkGraph.cpp)
#FIXME_NEXTSILICON: requires parallel_reduce on NextSilicon
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_CommonPolicyInterface.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_ExecutionSpace.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_Graph.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_MDRange_a.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_RangePolicy.cpp)
list(REMOVE_ITEM NextSilicon_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_Reductions.cpp)
kokkos_add_executable_and_test(
  CoreUnitTest_NextSilicon_SmokeTest SOURCES UnitTestMainInit.cpp ${NextSilicon_SOURCES_SMOKE}
)

# FIXME_NEXTSILICON: whitelist tests
set(NEXTSILICON_SOURCES
    ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_Abort.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_MinMaxClamp.cpp
    # ${CMAKE_CURRENT_BINARY_DIR}/nextsilicon/TestNextSilicon_DeepCopyAlignment.cpp # MDRange parallel for
)
kokkos_add_executable_and_test(CoreUnitTest_NextSilicon SOURCES UnitTestMainInit.cpp ${NEXTSILICON_SOURCES})
