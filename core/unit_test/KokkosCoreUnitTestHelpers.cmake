function(kokkos_core_unit_test_generate_backend_sources Tag)
  cmake_parse_arguments(PARSE_ARGV 1 KOKKOS_UNIT_TEST "" "HOST_ACCESSIBLE_CATEGORY" "")
  string(TOLOWER ${Tag} dir)

  set(dir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
  file(MAKE_DIRECTORY ${dir})
  # Needed to split this for Windows NVCC, since it ends up putting everything on the
  # command line in an intermediate compilation step even if CMake generated a response
  # file. That then exceeded the shell command line max length.
  set(${Tag}_SOURCES1A)
  foreach(
    Name
    ArrayOps
    AtomicOperations_complexdouble
    AtomicOperations_complexfloat
    AtomicOperations_double
    AtomicOperations_float
    AtomicOperations_int8
    AtomicOperations_int16
    AtomicOperations_int
    AtomicOperations_longint
    AtomicOperations_longlongint
    AtomicOperations_shared
    AtomicOperations_unsignedint
    AtomicOperations_unsignedlongint
    AtomicViews
    BitManipulationBuiltins
    CheckedIntegerOps
    CommonPolicyConstructors
    Concepts
    Crs
    DeepCopy_Narrowing
    DeepCopy_SameArgument
    DeepCopyAlignment
    ExecSpacePartitioning
    ExecSpaceThreadSafety
    FunctorAnalysis
    GraphNodeCtorProps
    HostSharedPtr
    HostSharedPtrAccessOnDevice
    JoinBackwardCompatibility
    LocalDeepCopy
    MathematicalConstants
    MathematicalFunctions1
    MathematicalFunctions2
    MathematicalFunctions3
    MathematicalSpecialFunctions
  )
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${Tag}_SOURCES1A ${file})
  endforeach()

  set(${Tag}_SOURCES1B)
  set(${Tag}_TESTNAMES1B
      MDRange_b
      MDRange_c
      MDRange_d
      MDRange_e
      MDRange_f
      MDRange_g
      MDRangePolicyConstructors
      MDRangeReduce
      MDSpan
      MDSpanAtomicAccessor
      MDSpanConversion
      MinMaxClamp
      NumericTraits
      OccupancyControlTrait
      Other
      ParallelScanRangePolicy
      CustomScalarParallelScan
      Printf
      QuadPrecisionMath
      RangePolicyConstructors
      RangePolicyRequire
      ReducerCTADs
      Reducers_a
      Reducers_b
      Reducers_c
      Reducers_d
      Reducers_e
      Reductions_DeviceView
      SpaceAwareAccessorAccessViolation
      SpaceAwareAccessor
      Swap
  )
  if(NOT Kokkos_ENABLE_IMPL_MDSPAN)
    list(REMOVE_ITEM ${Tag}_TESTNAMES1B MDSpanAtomicAccessor MDSpanConversion SpaceAwareAccessorAccessViolation
         SpaceAwareAccessor
    )
  endif()
  # This test case causes MSVC to fail with "number of sections exceeded object file format limit"
  if(MSVC)
    list(REMOVE_ITEM ${Tag}_TESTNAMES1B Reducers_d)
  endif()
  foreach(Name IN LISTS ${Tag}_TESTNAMES1B)
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${Tag}_SOURCES1B ${file})
  endforeach()

  # Add additional test(s) from subdirectory range_policy
  set(file ${dir}/Test${Tag}_RangePolicyExecutionTypes.cpp)
  file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n"
                              "#include <range_policy/TestRangePolicyExecutionTypes.hpp>\n"
  )
  configure_file(${dir}/dummy.cpp ${file})
  list(APPEND ${Tag}_SOURCES1B ${file})

  set(${Tag}_SOURCES2A)
  set(${Tag}_TESTNAMES2A
      TeamCombinedReducers
      TeamPolicyConstructors
      TeamReductionScan
      TeamScan
      TeamScratch
      TeamTeamSize
      Timer
      UniqueToken
      View_64bit
      ViewAPI_b
      ViewAPI_c
      ViewAPI_d
      ViewAPI_e
      ViewBadAlloc
      ViewCopy_a
      ViewCopy_b
      ViewCopy_c
      ViewCtorDimMatch
      ViewCtorProp
      ViewEmptyRuntimeUnmanaged
      ViewHooks
      ViewLayoutStrideAssignment
      ViewMapping_a
      ViewMapping_b
      ViewMapping_subview
      ViewMemoryAccessViolation
      ViewMove
      ViewOfClass
      ViewOfViews
      ViewOutOfBoundsAccess
      ViewResize
      WithoutInitializing
  )
  # Workaround to internal compiler error with intel classic compilers
  # when using -no-ip flag in ViewCopy_c
  # See issue: https://github.com/kokkos/kokkos/issues/7084
  if(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
    list(REMOVE_ITEM ${Tag}_TESTNAMES2A ViewCopy_c)
  endif()
  foreach(Name IN LISTS ${Tag}_TESTNAMES2A)
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${Tag}_SOURCES2A ${file})
  endforeach()

  if(KOKKOS_UNIT_TEST_HOST_ACCESSIBLE_CATEGORY)
    set(TagHostAccessible ${KOKKOS_UNIT_TEST_HOST_ACCESSIBLE_CATEGORY})
  else()
    set(TagHostAccessible ${Tag})
  endif()

  set(${Tag}_SOURCES2B)
  foreach(
    Name
    SubView_a
    SubView_b
    SubView_c01
    SubView_c02
    SubView_c03
    SubView_c04
    SubView_c05
  )
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${TagHostAccessible}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${Tag}_SOURCES2B ${file})
  endforeach()

  set(${Tag}_SOURCES2C)
  foreach(Name SubView_c06 SubView_c07 SubView_c08 SubView_c09)
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${TagHostAccessible}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${Tag}_SOURCES2C ${file})
  endforeach()

  set(${Tag}_SOURCES2D)
  foreach(
    Name
    SubView_c10
    SubView_c11
    SubView_c12
    SubView_c13
    SubView_c14
    SubView_c15
    SubView_c16
  )
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${TagHostAccessible}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${Tag}_SOURCES2D ${file})
  endforeach()

  set(${Tag}_SOURCES1 ${${Tag}_SOURCES1A} ${${Tag}_SOURCES1B})
  set(${Tag}_SOURCES2 ${${Tag}_SOURCES2A} ${${Tag}_SOURCES2B} ${${Tag}_SOURCES2C} ${${Tag}_SOURCES2D})
  set(${Tag}_SOURCES ${${Tag}_SOURCES1} ${${Tag}_SOURCES2})

  # ViewSupport should eventually contain the new implementation
  # detail tests for the mdspan based View
  if(Kokkos_ENABLE_IMPL_MDSPAN)
    set(BV_TestNames
        AllocationAndSpanSize
        BasicView
        CreateMirrorViewAndCopy
        LegacyLayoutFunction
        ReferenceCountedAccessor
        ReferenceCountedDataHandle
        ViewEqualityOperator
        ViewCtorConvertibleToPtr
    )
    if(NOT Kokkos_ENABLE_IMPL_VIEW_LEGACY)
      list(APPEND BV_TestNames ViewCustomizationAccessorArg ViewCustomizationAllocationType
           ViewCustomizationAccessorFromMapping MinimalViewMDSpanTemplateArgumentViability
      )
    endif()
    if(NOT Kokkos_ENABLE_IMPL_VIEW_LEGACY)
      list(APPEND BV_TestNames ViewCtorDataHandle)
    endif()
    set(${Tag}_VIEWSUPPORT)
    foreach(Name IN LISTS BV_TestNames)
      set(file ${dir}/Test${Tag}_View_${Name}.cpp)
      # Write to a temporary intermediate file and call configure_file to avoid
      # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
      file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <view/Test${Name}.hpp>\n")
      configure_file(${dir}/dummy.cpp ${file})
      list(APPEND ${Tag}_VIEWSUPPORT ${file})
    endforeach()

  endif()
  # Smoke test: minimal subset for quick iteration (generate sources not built with main targets).
  foreach(Name IN LISTS SMOKE_TEST_NAMES)
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    file(WRITE ${dir}/dummy.cpp "#include <Test${Tag}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
  endforeach()
  set(${Tag}_SOURCES_SMOKE)
  foreach(Name IN LISTS SMOKE_TEST_NAMES)
    set(file ${dir}/Test${Tag}_${Name}.cpp)
    list(APPEND ${Tag}_SOURCES_SMOKE ${file})
  endforeach()

  foreach(sources_name
          SOURCES1A
          SOURCES1B
          TESTNAMES1B
          SOURCES2A
          TESTNAMES2A
          SOURCES2B
          SOURCES2C
          SOURCES2D
          SOURCES1
          SOURCES2
          SOURCES
          VIEWSUPPORT
          SOURCES_SMOKE
  )
    set(${Tag}_${sources_name} ${${Tag}_${sources_name}} PARENT_SCOPE)
  endforeach()
endfunction()

function(kokkos_core_unit_test_generate_memory_space_sources DEVICE SPACE)
  string(TOLOWER ${DEVICE} dir)

  set(dir ${CMAKE_CURRENT_BINARY_DIR}/${dir})
  file(MAKE_DIRECTORY ${dir})
  foreach(
    Name
    SharedAlloc
    ViewAPI_a
    ViewAPI_b
    ViewAPI_c
    ViewAPI_d
    ViewAPI_e
    ViewCopy_a
    ViewCopy_b
    ViewCopy_c
    ViewMapping_a
    ViewMapping_b
    ViewMapping_subview
  )
    set(file ${dir}/Test${DEVICE}${SPACE}_${Name}.cpp)
    # Write to a temporary intermediate file and call configure_file to avoid
    # updating timestamps triggering unnecessary rebuilds on subsequent cmake runs.
    file(WRITE ${dir}/dummy.cpp "#include <Test${DEVICE}${SPACE}_Category.hpp>\n" "#include <Test${Name}.hpp>\n")
    configure_file(${dir}/dummy.cpp ${file})
    list(APPEND ${DEVICE}_SOURCES3 ${file})
    list(APPEND ${DEVICE}_SOURCES ${file})
  endforeach()

  set(${DEVICE}_SOURCES3 ${${DEVICE}_SOURCES3} PARENT_SCOPE)
  set(${DEVICE}_SOURCES ${${DEVICE}_SOURCES} PARENT_SCOPE)
endfunction()
