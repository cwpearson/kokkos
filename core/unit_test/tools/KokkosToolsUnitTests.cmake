if(KOKKOS_ENABLE_TUNING)
  kokkos_add_executable_and_test(CoreUnitTest_TuningBuiltins SOURCES UnitTestMainInit.cpp tools/TestBuiltinTuners.cpp)
  kokkos_add_executable_and_test(CoreUnitTest_TuningBasics SOURCES tools/TestTuning.cpp)
  kokkos_add_executable_and_test(CoreUnitTest_CategoricalTuner SOURCES tools/TestCategoricalTuner.cpp)
endif()

set(KOKKOSP_SOURCES UnitTestMainInit.cpp tools/TestEventCorrectness.cpp tools/TestKernelNames.cpp
                    tools/TestProfilingSection.cpp tools/TestScopedRegion.cpp tools/TestWithoutInitializing.cpp
)

# FIXME_NEXTSILICON: others require parallel_reduce and parallel_scan
if(KOKKOS_ENABLE_NEXTSILICON)
  set(KOKKOSP_SOURCES UnitTestMainInit.cpp tools/TestWithoutInitializing.cpp)
endif()

# if (NOT KOKKOS_ENABLE_NEXTSILICON)
kokkos_add_executable_and_test(CoreUnitTest_KokkosP SOURCES ${KOKKOSP_SOURCES})
# endif()
if(KOKKOS_ENABLE_LIBDL)
  kokkos_add_executable_and_test(CoreUnitTest_ToolIndependence SOURCES tools/TestIndependence.cpp)
  target_compile_definitions(Kokkos_CoreUnitTest_ToolIndependence PUBLIC KOKKOS_TOOLS_INDEPENDENT_BUILD)
  kokkos_add_test_library(kokkosprinter-tool SHARED SOURCES tools/printing-tool.cpp)

  if((NOT (Kokkos_ENABLE_CUDA AND WIN32)) AND (NOT ("${KOKKOS_CXX_COMPILER_ID}" STREQUAL "Fujitsu")))
    target_compile_features(kokkosprinter-tool PUBLIC cxx_std_14)
  endif()

  kokkos_add_test_library(kokkos-empty-profiling-tool SHARED SOURCES tools/empty-tool.cpp)

  kokkos_add_test_executable(ProfilingAllCalls tools/TestAllCalls.cpp)

  # FIXME_NEXTSILICON: requires parallel_reduce
  if(NOT KOKKOS_ENABLE_NEXTSILICON)
    kokkos_add_test_executable(ToolsInitialization UnitTestMain.cpp tools/TestToolsInitialization.cpp)
  endif()

  kokkos_add_test_executable(EmptyProfilingLibraryDeathTest UnitTestMain.cpp tools/TestEmptyProfilingLibrary.cpp)

  set(ADDRESS_REGEX "0x[0-9a-f]*")
  set(MEMSPACE_REGEX "[HC][ou][sd][ta][a-zA-Z]*")
  set(SIZE_REGEX "[0-9]*")
  set(SKIP_SCRATCH_INITIALIZATION_REGEX ".*")

  # check that loading a library with no profiling callbacks produces an error
  kokkos_add_test(
    SKIP_TRIBITS
    NAME
    ProfilingTestErrorNoCallbacks
    EXE
    EmptyProfilingLibraryDeathTest
    TOOL
    kokkos-empty-profiling-tool
  )

  # check help works via environment variable
  kokkos_add_test(
    SKIP_TRIBITS
    NAME
    ProfilingTestLibraryLoadHelp
    EXE
    ProfilingAllCalls
    TOOL
    kokkosprinter-tool
    ARGS
    --kokkos-tools-help
    PASS_REGULAR_EXPRESSION
    "kokkosp_init_library::kokkosp_print_help:Kokkos_ProfilingAllCalls::kokkosp_finalize_library::"
  )

  # check help works via direct library specification
  kokkos_add_test(
    SKIP_TRIBITS
    NAME
    ProfilingTestLibraryCmdLineHelp
    EXE
    ProfilingAllCalls
    ARGS
    --kokkos-tools-help
    --kokkos-tools-libs=$<TARGET_FILE:kokkosprinter-tool>
    PASS_REGULAR_EXPRESSION
    "kokkosp_init_library::kokkosp_print_help:Kokkos_ProfilingAllCalls::kokkosp_finalize_library::"
  )

  kokkos_add_test(
    SKIP_TRIBITS
    NAME
    ProfilingTestLibraryLoad
    EXE
    ProfilingAllCalls
    TOOL
    kokkosprinter-tool
    ARGS
    --kokkos-tools-args="-c test delimit"
    PASS_REGULAR_EXPRESSION
    "kokkosp_init_library::kokkosp_parse_args:4:Kokkos_ProfilingAllCalls:-c:test:delimit::.*::kokkosp_allocate_data:${MEMSPACE_REGEX}:source:${ADDRESS_REGEX}:40::kokkosp_begin_parallel_for:Kokkos::View::initialization [[]source] via memset:[0-9]+:0::kokkosp_end_parallel_for:0::kokkosp_allocate_data:${MEMSPACE_REGEX}:destination:${ADDRESS_REGEX}:40::kokkosp_begin_parallel_for:Kokkos::View::initialization [[]destination] via memset:[0-9]+:0::kokkosp_end_parallel_for:0::kokkosp_begin_deep_copy:${MEMSPACE_REGEX}:destination:${ADDRESS_REGEX}:${MEMSPACE_REGEX}:source:${ADDRESS_REGEX}:40::.*kokkosp_end_deep_copy::kokkosp_begin_parallel_for:parallel_for:${SIZE_REGEX}:0::kokkosp_end_parallel_for:0::kokkosp_begin_parallel_reduce:parallel_reduce:${SIZE_REGEX}:1${SKIP_SCRATCH_INITIALIZATION_REGEX}::kokkosp_end_parallel_reduce:1::kokkosp_begin_parallel_scan:parallel_scan:${SIZE_REGEX}:2::kokkosp_end_parallel_scan:2::kokkosp_push_profile_region:push_region::kokkosp_pop_profile_region::kokkosp_create_profile_section:created_section:3::kokkosp_start_profile_section:3::kokkosp_stop_profile_section:3::kokkosp_destroy_profile_section:3::kokkosp_profile_event:profiling_event::kokkosp_declare_metadata:dogs:good::kokkosp_deallocate_data:${MEMSPACE_REGEX}:destination:${ADDRESS_REGEX}:40::kokkosp_deallocate_data:${MEMSPACE_REGEX}:source:${ADDRESS_REGEX}:40::kokkosp_finalize_library::"
  )

  # Above will test that leading/trailing quotes are stripped bc ctest cmd args is:
  #       "--kokkos-tools-args="-c test delimit""
  # The bracket argument syntax: [=[ and ]=] used below ensures it is treated as
  # a single argument:
  #       "--kokkos-tools-args=-c test delimit"
  #
  # https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#bracket-argument
  #
  kokkos_add_test(
    SKIP_TRIBITS
    NAME
    ProfilingTestLibraryCmdLine
    EXE
    ProfilingAllCalls
    ARGS
    [=[--kokkos-tools-args=-c test delimit]=]
    --kokkos-tools-libs=$<TARGET_FILE:kokkosprinter-tool>
    PASS_REGULAR_EXPRESSION
    "kokkosp_init_library::kokkosp_parse_args:4:Kokkos_ProfilingAllCalls:-c:test:delimit::.*::kokkosp_allocate_data:${MEMSPACE_REGEX}:source:${ADDRESS_REGEX}:40::kokkosp_begin_parallel_for:Kokkos::View::initialization [[]source] via memset:[0-9]+:0::kokkosp_end_parallel_for:0::kokkosp_allocate_data:${MEMSPACE_REGEX}:destination:${ADDRESS_REGEX}:40::kokkosp_begin_parallel_for:Kokkos::View::initialization [[]destination] via memset:[0-9]+:0::kokkosp_end_parallel_for:0::kokkosp_begin_deep_copy:${MEMSPACE_REGEX}:destination:${ADDRESS_REGEX}:${MEMSPACE_REGEX}:source:${ADDRESS_REGEX}:40::.*kokkosp_end_deep_copy::kokkosp_begin_parallel_for:parallel_for:${SIZE_REGEX}:0::kokkosp_end_parallel_for:0::kokkosp_begin_parallel_reduce:parallel_reduce:${SIZE_REGEX}:1${SKIP_SCRATCH_INITIALIZATION_REGEX}::kokkosp_end_parallel_reduce:1::kokkosp_begin_parallel_scan:parallel_scan:${SIZE_REGEX}:2::kokkosp_end_parallel_scan:2::kokkosp_push_profile_region:push_region::kokkosp_pop_profile_region::kokkosp_create_profile_section:created_section:3::kokkosp_start_profile_section:3::kokkosp_stop_profile_section:3::kokkosp_destroy_profile_section:3::kokkosp_profile_event:profiling_event::kokkosp_declare_metadata:dogs:good::kokkosp_deallocate_data:${MEMSPACE_REGEX}:destination:${ADDRESS_REGEX}:40::kokkosp_deallocate_data:${MEMSPACE_REGEX}:source:${ADDRESS_REGEX}:40::kokkosp_finalize_library::"
  )
endif() #KOKKOS_ENABLE_LIBDL
