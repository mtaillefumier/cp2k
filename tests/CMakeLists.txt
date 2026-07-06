# Registers CP2K's regression test suite (TEST_DIRS + UNIT_TESTS) as native
# CTest tests -- one add_test() per individual input file / unit test.
#
# The actual list of tests is produced by generate_ctest_list.py, run once
# here at configure time (see execute_process below) rather than written by
# hand, since it tracks TEST_DIRS/TEST_FILES.toml directly. Requirement
# filtering (feature flags, mpiranks conditions) happens at `ctest` run time
# instead of here, because those flags only exist once this build's binary
# has actually been built and run -- see ctest_run_one.py and
# ctest_prepare_environment.py.

set(CP2K_REGTEST_MPIRANKS
    "2" CACHE STRING "Number of MPI ranks used for regtests that require MPI")
set(CP2K_REGTEST_OMPTHREADS
    "0" CACHE STRING
    "OpenMP threads per regtest (0 = auto: 2 for *smp builds, else 1)")
set(CP2K_REGTEST_TIMEOUT
    "400" CACHE STRING "Per-test timeout in seconds")
set(CP2K_REGTEST_MPIEXEC
    "mpiexec -n {N} --bind-to none" CACHE STRING
    "Template for launching MPI regtests, {N} is replaced by the rank count")
set(CP2K_REGTEST_NUM_GPUS
    "0" CACHE STRING
    "Number of GPU devices to round-robin regtests across (0 = auto-detect)")
option(CP2K_REGTEST_VALGRIND "Run regtests under Valgrind memcheck" OFF)
set(CP2K_REGTEST_RESTRICT_DIR
    "" CACHE STRING
    "Semicolon-separated regexes; only matching TEST_DIRS/UNIT_TESTS entries \
are registered (empty = all)")
mark_as_advanced(
  CP2K_REGTEST_MPIRANKS CP2K_REGTEST_OMPTHREADS CP2K_REGTEST_TIMEOUT
  CP2K_REGTEST_MPIEXEC CP2K_REGTEST_NUM_GPUS CP2K_REGTEST_VALGRIND
  CP2K_REGTEST_RESTRICT_DIR)

set(CP2K_REGTESTING_DIR "${CMAKE_CURRENT_BINARY_DIR}/regtesting_data")
set(CP2K_REGTEST_FLAGS_FILE "${CMAKE_CURRENT_BINARY_DIR}/cp2kflags.txt")
set(CP2K_REGTEST_GENERATED_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/generated_regtests.cmake")

set(_cp2k_regtest_extra_args "")
if(CP2K_REGTEST_VALGRIND)
  list(APPEND _cp2k_regtest_extra_args --valgrind)
endif()
foreach(_cp2k_regtest_pattern IN LISTS CP2K_REGTEST_RESTRICT_DIR)
  list(APPEND _cp2k_regtest_extra_args --restrict-dir "${_cp2k_regtest_pattern}")
endforeach()

execute_process(
  COMMAND
    "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_ctest_list.py"
    --cp2k-root "${PROJECT_SOURCE_DIR}"
    --tests-source-dir "${CMAKE_CURRENT_SOURCE_DIR}"
    --regtesting-dir "${CP2K_REGTESTING_DIR}"
    --flags-file "${CP2K_REGTEST_FLAGS_FILE}"
    --binary-dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    --version "${__cp2k_ext}"
    --output "${CP2K_REGTEST_GENERATED_FILE}"
    --mpiranks "${CP2K_REGTEST_MPIRANKS}"
    --ompthreads "${CP2K_REGTEST_OMPTHREADS}"
    --timeout "${CP2K_REGTEST_TIMEOUT}"
    --mpiexec "${CP2K_REGTEST_MPIEXEC}"
    --num-gpus "${CP2K_REGTEST_NUM_GPUS}"
    ${_cp2k_regtest_extra_args}
  RESULT_VARIABLE _cp2k_regtest_generate_result
  OUTPUT_VARIABLE _cp2k_regtest_generate_stdout
  ERROR_VARIABLE _cp2k_regtest_generate_stderr)

if(NOT _cp2k_regtest_generate_result EQUAL 0)
  message(
    FATAL_ERROR
    "Failed to generate the regression test list:\n"
    "${_cp2k_regtest_generate_stdout}\n${_cp2k_regtest_generate_stderr}")
endif()
message(STATUS "${_cp2k_regtest_generate_stderr}")

include("${CP2K_REGTEST_GENERATED_FILE}")

file(READ "${CMAKE_CURRENT_BINARY_DIR}/mpi_prefix.txt" _cp2k_regtest_mpi_prefix)
string(STRIP "${_cp2k_regtest_mpi_prefix}" _cp2k_regtest_mpi_prefix)

add_test(
  NAME regtest_environment_setup
  COMMAND
    "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/ctest_prepare_environment.py"
    --cp2k-root "${PROJECT_SOURCE_DIR}"
    --regtesting-dir "${CP2K_REGTESTING_DIR}"
    --binary-dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    --version "${__cp2k_ext}"
    --mpi-prefix "${_cp2k_regtest_mpi_prefix}"
    --flags-out "${CP2K_REGTEST_FLAGS_FILE}")
set_tests_properties(regtest_environment_setup PROPERTIES
    FIXTURES_SETUP regtest_environment_setup)

# Regenerate the test list whenever the test definitions change, without
# requiring a manual `cmake` rerun.
file(GLOB_RECURSE _cp2k_regtest_toml_files
     "${CMAKE_CURRENT_SOURCE_DIR}/*/TEST_FILES.toml")
set_property(
  DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/TEST_DIRS"
  "${CMAKE_CURRENT_SOURCE_DIR}/UNIT_TESTS"
  ${_cp2k_regtest_toml_files})

# EOF
