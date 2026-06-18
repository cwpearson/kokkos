kokkos_add_executable_and_test(
  CoreUnitTest_DefaultInstanceFencesOnFinalize SOURCES UnitTestMain.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/default/TestDefaultInstanceFencesOnFinalize.cpp
)

set(DEFAULT_DEVICE_SOURCES
    UnitTestMainInit.cpp
    TestCStyleMemoryManagement.cpp
    TestSharedSpace.cpp
    TestSharedHostPinnedSpace.cpp
    TestCompilerMacros.cpp
    default/TestDefaultDeviceType.cpp
    default/TestDefaultDeviceType_a1.cpp
    default/TestDefaultDeviceType_b1.cpp
    default/TestDefaultDeviceType_c1.cpp
    default/TestDefaultDeviceType_a2.cpp
    default/TestDefaultDeviceType_b2.cpp
    default/TestDefaultDeviceType_c2.cpp
    default/TestDefaultDeviceType_a3.cpp
    default/TestDefaultDeviceType_b3.cpp
    default/TestDefaultDeviceType_c3.cpp
    default/TestDefaultDeviceTypeResize.cpp
    default/TestDefaultDeviceTypeViewAPI.cpp
)
# FIXME_OPENACC do not provide a MemorySpace that can be accessed from all ExecSpaces
# FIXME_SYCL clock_tic does not give the correct timings for cloc_tic
# FIXME_NEXTSILICON requires parallel_reduce
if(KOKKOS_ENABLE_OPENACC OR KOKKOS_ENABLE_SYCL)
  list(REMOVE_ITEM DEFAULT_DEVICE_SOURCES TestSharedSpace.cpp)
endif()
# FIXME_OPENACC do not provide a HostPinnedMemorySpace that can be accessed from all ExecSpaces
# FIXME_NEXTSILICON: does not provide a HostPinnedMemorySpace
if(KOKKOS_ENABLE_OPENACC OR KOKKOS_ENABLE_NEXTSILICON)
  list(REMOVE_ITEM DEFAULT_DEVICE_SOURCES TestSharedHostPinnedSpace.cpp)
endif()

# FIXME_OPENACC - Comment non-passing tests with the NVIDIA HPC compiler nvc++
if(KOKKOS_ENABLE_OPENACC AND KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  list(
    REMOVE_ITEM
    DEFAULT_DEVICE_SOURCES
    default/TestDefaultDeviceType_a1.cpp
    default/TestDefaultDeviceType_b1.cpp
    default/TestDefaultDeviceType_c1.cpp
    default/TestDefaultDeviceType_a2.cpp
    default/TestDefaultDeviceType_b2.cpp
    default/TestDefaultDeviceType_c2.cpp
    default/TestDefaultDeviceType_a3.cpp
    default/TestDefaultDeviceType_b3.cpp
    default/TestDefaultDeviceType_c3.cpp
    default/TestDefaultDeviceTypeResize.cpp
    default/TestDefaultDeviceTypeViewAPI.cpp
  )
endif()

# FIXME_OPENACC - Comment non-passing tests with the Clang compiler
if(KOKKOS_ENABLE_OPENACC AND KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  list(
    REMOVE_ITEM
    DEFAULT_DEVICE_SOURCES
    default/TestDefaultDeviceType_a1.cpp
    default/TestDefaultDeviceType_b1.cpp
    default/TestDefaultDeviceType_c1.cpp
    default/TestDefaultDeviceType_a2.cpp
    default/TestDefaultDeviceType_b2.cpp
    default/TestDefaultDeviceType_c2.cpp
    default/TestDefaultDeviceType_a3.cpp
    default/TestDefaultDeviceType_b3.cpp
    default/TestDefaultDeviceType_c3.cpp
    default/TestDefaultDeviceTypeResize.cpp
    default/TestDefaultDeviceTypeViewAPI.cpp
  )
endif()

#FIXME_NEXTSILICON: requires parallel_for
if(NOT KOKKOS_ENABLE_NEXTSILICON)
  kokkos_add_executable_and_test(CoreUnitTest_Default SOURCES ${DEFAULT_DEVICE_SOURCES})
endif()

# FIXME_NEXTSILICON: requires parallel_reduce
if(NOT Kokkos_ENABLE_NEXTSILICON)
  kokkos_add_executable_and_test(
    CoreUnitTest_InitializeFinalize
    SOURCES
    UnitTestMain.cpp
    TestExecutionEnvironmentNonInitializedOrFinalized.cpp
    TestInitializationSettings.cpp
    TestInitializeFinalize.cpp
    TestKokkosHelpCausesNormalProgramTermination.cpp
    TestLegionInitialization.cpp
    TestParseCmdLineArgsAndEnvVars.cpp
    TestPushFinalizeHook.cpp
    TestScopeGuard.cpp
  )
endif()

# This test is intended for development and debugging by putting code
# into TestDefaultDeviceDevelop.cpp. By default its empty.
kokkos_add_executable_and_test(CoreUnitTest_Develop SOURCES UnitTestMainInit.cpp default/TestDefaultDeviceDevelop.cpp)
