#
# Define Incremental Testing Feature Levels
# Define Device name mappings (i.e. what comes after Kokkos:: for the ExecSpace)
#

set(KOKKOS_CUDA_FEATURE_LEVEL 999)
set(KOKKOS_CUDA_NAME Cuda)
set(KOKKOS_HIP_FEATURE_LEVEL 999)
set(KOKKOS_HIP_NAME HIP)
set(KOKKOS_HPX_FEATURE_LEVEL 999)
set(KOKKOS_HPX_NAME Experimental::HPX)
set(KOKKOS_OPENMP_FEATURE_LEVEL 999)
set(KOKKOS_OPENMP_NAME OpenMP)
set(KOKKOS_SERIAL_FEATURE_LEVEL 999)
set(KOKKOS_SERIAL_NAME Serial)
set(KOKKOS_SYCL_FEATURE_LEVEL 999)
set(KOKKOS_SYCL_NAME SYCL)
set(KOKKOS_THREADS_FEATURE_LEVEL 999)
set(KOKKOS_THREADS_NAME Threads)
# FIXME_OPENACC - The Clang compiler only compiles the first 9 incremental tests for the OpenACC backend.
if(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  set(KOKKOS_OPENACC_FEATURE_LEVEL 9)
else()
  set(KOKKOS_OPENACC_FEATURE_LEVEL 17)
endif()

set(KOKKOS_NEXTSILICON_FEATURE_LEVEL 4)
set(KOKKOS_NEXTSILICON_NAME Experimental::NextSilicon)

set(KOKKOS_OPENACC_NAME Experimental::OpenACC)

#
# Define the tests
#

#I will leave these alone for now because I don't need transitive dependencies on tests
kokkos_include_directories(${CMAKE_CURRENT_BINARY_DIR})
kokkos_include_directories(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
kokkos_include_directories(${KOKKOS_SOURCE_DIR}/core/unit_test/category_files)

include(KokkosCoreUnitTestHelpers.cmake)

set(COMPILE_ONLY_SOURCES
    TestArray.cpp
    TestCreateMirror.cpp
    TestDetectionIdiom.cpp
    TestBitManipulation.cpp
    TestFunctionAnnotationMacros.cpp
    TestInterOp.cpp
    TestRangePolicyCTAD.cpp
    TestStringManipulation.cpp
    TestVersionMacros.cpp
    TestViewRank.cpp
    TestViewTypeTraits.cpp
    TestViewTypedefs.cpp
    TestTypeInfo.cpp
    TestTypeList.cpp
    TestMDRangePolicyCTAD.cpp
    TestTeamPolicyCTAD.cpp
    TestTeamMDRangePolicyCTAD.cpp
    TestNestedReducerCTAD.cpp
    view/TestAccessorFromMemoryTraits.cpp
    view/TestBasicViewMDSpanConversion.cpp
    view/TestViewCustomization.cpp
    view/TestExtentsDatatypeConversion.cpp
    view/TestConversionFromPointer.cpp
    view/TestMemoryTraitTypes.cpp
    view/TestViewMDSpanTemplateArgumentEquivalence.cpp
)

include(nextsilicon/KokkosNextSiliconCompileOnlyTests.cmake)

if(NOT Kokkos_ENABLE_IMPL_MDSPAN OR KOKKOS_CXX_COMPILER_ID STREQUAL "Intel")
  list(REMOVE_ITEM COMPILE_ONLY_SOURCES view/TestBasicViewMDSpanConversion.cpp)
endif()

if(NOT Kokkos_ENABLE_IMPL_MDSPAN OR Kokkos_ENABLE_IMPL_VIEW_LEGACY)
  list(REMOVE_ITEM COMPILE_ONLY_SOURCES view/TestAccessorFromMemoryTraits.cpp)
  list(REMOVE_ITEM COMPILE_ONLY_SOURCES view/TestViewCustomization.cpp)
  list(REMOVE_ITEM COMPILE_ONLY_SOURCES view/TestViewMDSpanTemplateArgumentEquivalence.cpp)
endif()

#testing if windows.h and Kokkos_Core.hpp can be included
if(WIN32)
  list(APPEND COMPILE_ONLY_SOURCES TestWindowsInclude.cpp)
endif()

if(Kokkos_INSTALL_TESTING) # FIXME Kokkos_ and KOKKOS_ variables are out of sync
  if(Kokkos_CXX_COMPILER_ID STREQUAL "NVIDIA" AND Kokkos_CXX_HOST_COMPILER_ID STREQUAL "MSVC")
    list(REMOVE_ITEM COMPILE_ONLY_SOURCES TestTypeInfo.cpp)
  endif()
else()
  if(KOKKOS_CXX_COMPILER_ID STREQUAL "NVIDIA" AND KOKKOS_CXX_HOST_COMPILER_ID STREQUAL "MSVC")
    list(REMOVE_ITEM COMPILE_ONLY_SOURCES TestTypeInfo.cpp)
  endif()
endif()

kokkos_add_executable(CoreTestCompileOnly SOURCES TestCompileMain.cpp ${COMPILE_ONLY_SOURCES})

# Smoke tests, one per major API category
set(SMOKE_TEST_NAMES
    Abort
    Atomics
    BlockSizeDeduction
    CommonPolicyInterface
    Complex
    DeepCopy_Assignment
    ExecutionSpace
    Graph
    MDRange_a
    RangePolicy
    Reductions
    SharedAlloc
    TeamBasic
    TeamMDRange
    TeamVectorRange
    ViewAPI_a
    WorkGraph
)

include(default/KokkosDefaultDeviceUnitTests.cmake)
if(Kokkos_ENABLE_SERIAL)
  include(serial/KokkosSerialUnitTests.cmake)
endif()
if(Kokkos_ENABLE_THREADS)
  include(threads/KokkosThreadsUnitTests.cmake)
endif()
if(Kokkos_ENABLE_OPENMP)
  include(openmp/KokkosOpenMPUnitTests.cmake)
endif()
if(Kokkos_ENABLE_HPX)
  include(hpx/KokkosHPXUnitTests.cmake)
endif()
if(Kokkos_ENABLE_OPENACC)
  include(openacc/KokkosOpenACCUnitTests.cmake)
endif()
if(Kokkos_ENABLE_CUDA)
  include(cuda/KokkosCudaUnitTests.cmake)
endif()
if(Kokkos_ENABLE_HIP)
  include(hip/KokkosHIPUnitTests.cmake)
endif()
if(Kokkos_ENABLE_SYCL)
  include(sycl/KokkosSYCLUnitTests.cmake)
endif()
if(Kokkos_ENABLE_NEXTSILICON)
  include(nextsilicon/KokkosNextSiliconUnitTests.cmake)
endif()
include(tools/KokkosToolsUnitTests.cmake)

kokkos_add_test_executable(
  StackTraceTestExec
  SOURCES
  TestStackTrace.cpp
  TestStackTrace_f0.cpp
  TestStackTrace_f1.cpp
  TestStackTrace_f2.cpp
  TestStackTrace_f3.cpp
  TestStackTrace_f4.cpp
)
# We need -rdynamic on GNU platforms for the stacktrace functionality
# to work correctly with shared libraries
kokkos_set_exe_property(StackTraceTestExec ENABLE_EXPORTS ON)

kokkos_add_test(NAME CoreUnitTest_StackTraceTest EXE StackTraceTestExec FAIL_REGULAR_EXPRESSION "FAILED")

if(KOKKOS_ENABLE_HWLOC)
  kokkos_add_executable_and_test(CoreUnitTest_HWLOC SOURCES UnitTestMain.cpp TestHWLOC.cpp)
endif()

function(KOKKOS_ADD_INCREMENTAL_TEST DEVICE)
  kokkos_option(${DEVICE}_EXCLUDE_TESTS "" STRING "Incremental test exclude list")
  # Add unit test main
  set(${DEVICE}_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/UnitTestMainInit.cpp)

  # Iterate over incremental tests in directory

  append_glob(INCREMENTAL_FILE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/incremental/*.hpp)

  set(DEVICE_NAME ${KOKKOS_${DEVICE}_NAME})
  foreach(CURRENT_FILE_PATH ${INCREMENTAL_FILE_LIST})
    get_filename_component(CURRENT_FILE_NAME ${CURRENT_FILE_PATH} NAME)
    string(REPLACE ".hpp" "" CURRENT_TEST_NAME ${CURRENT_FILE_NAME})
    if(NOT CURRENT_TEST_NAME IN_LIST Kokkos_${DEVICE}_EXCLUDE_TESTS)
      set(CURRENT_TEST_OUTPUT_FILENAME ${CURRENT_TEST_NAME}_${DEVICE})
      file(STRINGS ${CURRENT_FILE_PATH} CURRENT_REQUIRED_FEATURE_LINE REGEX "Kokkos_Feature_Level_Required")
      # From each test get level implementation required
      string(REGEX REPLACE ".*Kokkos_Feature_Level_Required:" "" CURRENT_REQUIRED_FEATURE_LEVEL
                           ${CURRENT_REQUIRED_FEATURE_LINE}
      )
      # Cross-reference list of dependencies with selected feature list > matching feature test files are added to test applications
      if(KOKKOS_${DEVICE}_FEATURE_LEVEL GREATER_EQUAL CURRENT_REQUIRED_FEATURE_LEVEL)
        configure_file(
          IncrementalTest.cpp.in ${CMAKE_BINARY_DIR}/core/unit_test/generated/${CURRENT_TEST_OUTPUT_FILENAME}.cpp
        )
        set(${DEVICE}_SOURCES ${${DEVICE}_SOURCES};
                              ${CMAKE_BINARY_DIR}/core/unit_test/generated/${CURRENT_TEST_OUTPUT_FILENAME}.cpp
        )
      endif()
    endif()
  endforeach()

  string(TOUPPER ${DEVICE} UC_DEVICE)

  kokkos_option(ENABLE_${UC_DEVICE} ON BOOL "ENABLE ${UC_DEVICE}")

  kokkos_add_executable_and_test(IncrementalTest_${DEVICE} SOURCES ${${DEVICE}_SOURCES})

  set(EXE_NAME ${PACKAGE_NAME}_IncrementalTest_${DEVICE})
  # Check that the target was actually created because in a TribITS build
  # where only tests marked as PERFORMANCE enabled it would not be.
  if(TARGET ${EXE_NAME})
    target_include_directories(${EXE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/incremental)
  endif()

endfunction()

foreach(DEVICE ${KOKKOS_ENABLED_DEVICES})
  kokkos_add_incremental_test(${DEVICE})
endforeach()

kokkos_add_executable_and_test(CoreUnitTest_CTestDevice SOURCES UnitTestMain.cpp TestCTestDevice.cpp)

kokkos_add_executable_and_test(
  CoreUnitTest_CMakePassCmdLineArgs SOURCES UnitTest_CMakePassCmdLineArgs.cpp ARGS "one 2 THREE"
)

kokkos_add_executable(CoreUnitTest_CMakeTriBITSCompatibility SOURCES UnitTest_CMakeTriBITSCompatibility.cpp)

kokkos_add_test(NAME CoreUnitTest_CMakeTriBITSCompatibilityWillFail EXE CoreUnitTest_CMakeTriBITSCompatibility)
set_property(TEST Kokkos_CoreUnitTest_CMakeTriBITSCompatibilityWillFail PROPERTY WILL_FAIL TRUE)

set(Kokkos_CoreUnitTest_CMakeTriBITSCompatibilityDisable_DISABLE ON)
kokkos_add_test(NAME CoreUnitTest_CMakeTriBITSCompatibilityDisable EXE CoreUnitTest_CMakeTriBITSCompatibility)

set(Kokkos_CoreUnitTest_CMakeTriBITSCompatibilityExtraArgs_EXTRA_ARGS "--kokkos-test-tribits-compatibility=1")
kokkos_add_test(NAME CoreUnitTest_CMakeTriBITSCompatibilityExtraArgs EXE CoreUnitTest_CMakeTriBITSCompatibility)

set(Kokkos_CoreUnitTest_CMakeTriBITSCompatibilityEnvironment_ENVIRONMENT "KOKKOS_TEST_TRIBITS_COMPATIBILITY=1")
kokkos_add_test(NAME CoreUnitTest_CMakeTriBITSCompatibilityEnvironment EXE CoreUnitTest_CMakeTriBITSCompatibility)

set_source_files_properties(UnitTest_DeviceAndThreads.cpp PROPERTIES LANGUAGE ${KOKKOS_COMPILE_LANGUAGE})
add_executable(Kokkos_CoreUnitTest_DeviceAndThreads UnitTest_DeviceAndThreads.cpp)
target_link_libraries(Kokkos_CoreUnitTest_DeviceAndThreads Kokkos::kokkoscore)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND AND Python3_VERSION VERSION_GREATER_EQUAL 3.7)
  file(
    GENERATE
    OUTPUT $<TARGET_FILE_DIR:Kokkos_CoreUnitTest_DeviceAndThreads>/TestDeviceAndThreads.py
    INPUT TestDeviceAndThreads.py
    USE_SOURCE_PERMISSIONS
  )
  add_test(NAME Kokkos_CoreUnitTest_DeviceAndThreads
           COMMAND ${Python3_EXECUTABLE}
                   $<TARGET_FILE_DIR:Kokkos_CoreUnitTest_DeviceAndThreads>/TestDeviceAndThreads.py -v
  )
endif()

if(KOKKOS_ENABLE_HEADER_SELF_CONTAINMENT_TESTS AND NOT WIN32)
  add_subdirectory(headers_self_contained)
endif()
