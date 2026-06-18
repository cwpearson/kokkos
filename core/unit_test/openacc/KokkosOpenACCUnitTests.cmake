kokkos_core_unit_test_generate_backend_sources(OpenACC)

if(Kokkos_ENABLE_IMPL_MDSPAN)
  list(
    REMOVE_ITEM
    OpenACC_VIEWSUPPORT
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_View_ViewCustomizationAccessorArg.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_View_ViewCustomizationAllocationType.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_View_ViewCustomizationAccessorFromMapping.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_View_MinimalViewMDSpanTemplateArgumentViability.cpp
  )
  kokkos_add_executable_and_test(CoreUnitTest_OpenACC_ViewSupport SOURCES UnitTestMainInit.cpp ${OpenACC_VIEWSUPPORT})
endif()

# Same as OpenACC_SOURCES (WorkGraph removed from the main OpenACC unit-test bundle).
list(REMOVE_ITEM OpenACC_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_WorkGraph.cpp)
list(REMOVE_ITEM OpenACC_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamBasic.cpp)
list(REMOVE_ITEM OpenACC_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamMDRange.cpp)
list(REMOVE_ITEM OpenACC_SOURCES_SMOKE ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamVectorRange.cpp)
kokkos_add_executable_and_test(CoreUnitTest_OpenACC_SmokeTest SOURCES UnitTestMainInit.cpp ${OpenACC_SOURCES_SMOKE})

list(
  REMOVE_ITEM
  OpenACC_SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_complexdouble.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_complexfloat.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Crs.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_JoinBackwardCompatibility.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_LocalDeepCopy.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Other.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamCombinedReducers.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamMDRange.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamReductionScan.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamScan.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewAPI_e.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewMapping_subview.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewOfClass.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_WorkGraph.cpp
)

# FIXME_OPENACC - Comment non-passing tests with the NVIDIA HPC compiler nvc++
if(KOKKOS_CXX_COMPILER_ID STREQUAL NVHPC)
  list(
    REMOVE_ITEM
    OpenACC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/default/TestDefaultDeviceType_a1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/default/TestDefaultDeviceType_b1.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_shared.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_BlockSizeDeduction.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_DeepCopyAlignment.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_HostSharedPtr.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_HostSharedPtrAccessOnDevice.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalFunctions1.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalFunctions2.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalFunctions3.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MDRange_f.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_NumericTraits.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_RangePolicy.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_RangePolicyRequire.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reducers_d.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reductions_DeviceView.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamBasic.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamScratch.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamTeamSize.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_UniqueToken.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewResize.cpp
  )
endif()

# FIXME_OPENACC - Comment non-passing tests with the Clang compiler
if(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  list(
    REMOVE_ITEM
    OpenACC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/default/TestDefaultDeviceType_a1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/default/TestDefaultDeviceType_b1.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_double.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_float.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_int.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_int8.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_int16.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_longint.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_longlongint.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_shared.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_unsignedint.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicOperations_unsignedlongint.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Atomics.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_AtomicViews.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_BlockSizeDeduction.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_DeepCopyAlignment.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_HostSharedPtrAccessOnDevice.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalFunctions1.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalFunctions2.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MDRange_f.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_NumericTraits.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_RangePolicy.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_RangePolicyRequire.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reducers_a.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reducers_c.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reducers_d.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reductions.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Reductions_DeviceView.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamBasic.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamScratch.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_TeamTeamSize.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_UniqueToken.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewMapping_b.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewResize.cpp
    # This test is not removed above for OpenACC+NVHPC but all its TEST
    # functions are not compiled for the case of KOKKOS_COMPILER_NVHPC.
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewCtorDimMatch.cpp
    # These tests are not removed above for OpenACC+NVHPC.
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Abort.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_Complex.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ExecutionSpace.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalConstants.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalSpecialFunctions.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MinMaxClamp.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewLayoutStrideAssignment.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewMapping_a.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewMemoryAccessViolation.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_WithoutInitializing.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewAPI_d.cpp
    #Below test is disabled because it uses atomic operations not supported by Clacc.
    ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ExecSpaceThreadSafety.cpp
  )
  # When tested on a systme with AMD MI60 GPU and ROCm V5.4.0, these cause
  # clang-linker-wrapper to hang for a long time while building the unit tests.
  # In some cases, including them caused the build not to complete after an hour,
  # but excluding them permitted the build to finish in 1.5 mins or less.
  if(KOKKOS_ARCH_VEGA) # FIXME_OPENACC
    list(
      REMOVE_ITEM
      OpenACC_SOURCES
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_BitManipulationBuiltins.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_MathematicalFunctions3.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ParallelScanRangePolicy.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_CustomScalarParallelScan.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c04.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c05.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c06.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c07.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c08.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c09.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c10.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c11.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_SubView_c12.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewAPI_b.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/openacc/TestOpenACC_ViewAPI_c.cpp
    )
  endif()
endif()

kokkos_add_executable_and_test(CoreUnitTest_OpenACC SOURCES UnitTestMainInit.cpp ${OpenACC_SOURCES})
