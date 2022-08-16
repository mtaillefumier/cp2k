cmake_minimum_required(VERSION 3.22)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# =================================================================================================
# REQUIRE OUT-OF-SOURCE BUILDS
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(
    FATAL_ERROR
    "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory."
    )
endif()

# =================================================================================================
# PROJECT AND VERSION
include(GetGitRevisionDescription)
include(GNUInstallDirs)

git_describe(GIT_DESC)

if(GIT_DESC)
  string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR
    "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH
    "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_GIT
    "${GIT_DESC}")

  git_local_changes(GIT_STATE)
  if("${GIT_STATE}" STREQUAL "DIRTY")
    set(VERSION_GIT "${VERSION_GIT}-dirty")
  endif()

  execute_process(
    COMMAND git log -1 --format=%ai
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # take only the date from the git timestamp:
  string(REGEX REPLACE "^([0-9\\-]+) .*" "\\1" VERSION_DATE
    "${GIT_COMMIT_DATE}")
else()
  file(STRINGS VERSION VERSION_INFO)
  foreach(line ${VERSION_INFO})
    if(${line} MATCHES "^([^#].*)=[ \t]*(.*)$")
      set(key ${CMAKE_MATCH_1})
      set(value ${CMAKE_MATCH_2})
      string(REGEX REPLACE "[ \t\n]+$" "" key "${key}")
      string(REGEX REPLACE "[ \t\n]+$" "" value "${value}")
      set(VERSION_${key} "${value}")
      continue()
    endif()
  endforeach()
endif()

project(
  cp2k
  DESCRIPTION "CP2K"
  LANGUAGES Fortran C CXX)
set(cp2k_VERSION
  ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_GIT})
set(cp2k_APIVERSION ${VERSION_MAJOR}.${VERSION_MINOR})

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig)


# ##############################################################################
# Define the paths for static libraries and executables
# ##############################################################################
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${cp2k_BINARY_DIR}/lib
  CACHE PATH "Single output directory for building all libraries.")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${cp2k_BINARY_DIR}/bin
  CACHE PATH "Single output directory for building all executables.")

# Search for rocm in common locations

list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

foreach(__var ROCM_ROOT CRAY_ROCM_ROOT ORNL_ROCM_ROOT)
  if(ENV{${__var}})
    list(APPEND CMAKE_PREFIX_PATH ENV{__var})
    set(ROCM_PATH
      $ENV{__var}
      CACHE PATH "Path to ROCm installation")
  endif()
endforeach()


# =================================================================================================
# OPTIONS
include(CMakeDependentOption)

# make MPI optional as well. More work but might be useful
#set(CP2K_USE_MPI ON CACHE BOOL "Enable MPI support")

set(CP2K_SET_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable position independent code")

set(CP2K_USE_SIRIUS
  OFF
  CACHE BOOL "Enable plane wave dft calculations with sirius")
set(CP2K_USE_FFTW3
  ON
  CACHE BOOL "Use fftw3 for the calculating fast fourier transforms")
set(CP2K_USE_ELPA
  OFF
  CACHE BOOL "Enable elpa support")
set(CP2K_USE_PTSCOTCH
  OFF
  CACHE BOOL "Enable ptscotch support")
set(CP2K_USE_COSMA
  ON
  CACHE BOOL "COSMA is a drop in replacement of scalapack dgemm")
set(CP2K_USE_LIBINT2
  ON
  CACHE BOOL "Enable libint2 support")
set(CP2K_USE_VORI
  OFF
  CACHE BOOL "Enable libvori support")
set(CP2K_USE_PEXSI
  OFF
  CACHE BOOL "Enable pexsi support")
set(CP2K_USE_QUIP
  OFF
  CACHE BOOL "Enable quip support")
set(CP2K_USE_SPGLIB
  ON
  CACHE BOOL "Enable spglib support")
set(CP2K_USE_LIBXC
  ON
  CACHE BOOL "Enable libxc support")
set(CP2K_USE_SCALAPACK
  ON
  CACHE BOOL "Enable scalapack support")
set(CP2K_USE_STATIC_BLAS
  OFF
  CACHE BOOL "Link against static version of BLAS/LAPACK")
set(CP2K_USE_SPLA
  OFF
  CACHE BOOL
  "Use SPLA offloading gemm feature to the GPU if it is beneficial. ")

set(CP2K_USE_LIBXSMM
  ON
  CACHE BOOL "Use libxsmm for small gemms (supports x86 platforms)")
set(SCALAPACK_VENDOR_LIST "MKL" "SCI" "GENERIC")

set(SCALAPACK_VENDOR
  "GENERIC"
  CACHE STRING "scalapack vendor/generic backend")
set_property(CACHE SCALAPACK_VENDOR PROPERTY STRINGS ${SCALAPACK_VENDOR_LIST})

if(NOT ${SCALAPACK} IN_LIST SCALAPACK_VENDOR_LIST)
  message(FATAL_ERROR "Invalid scalapack vendor backend")
endif()

set(SUPPORTED_ACCELEARTION_TARGETS CUDA HIP NONE)
set(SUPPORTED_CUDA_ARCHITECTURES K20X K40 K80 P100 V100 A100)
set(SUPPORTED_HIP_ARCHITECTURES
  Mi50
  Mi100
  Mi210
  Mi250X
  K20X
  K40
  K80
  P100
  V100
  A100)
set(WITH_GPU
  "P100"
  CACHE STRING
  "Set the CUDA GPU architecture if HIP is enabled (default: P100)")

set_property(CACHE WITH_GPU PROPERTY STRINGS ${SUPPORTED_CUDA_ARCHITECTURES}
  ${SUPPORTED_HIP_ARCHITECTURES})

set(CP2K_USE_ACCEL "NONE"
  CACHE STRING "Set hardware acceleartion support: CUDA, HIP")

set_property(CACHE CP2K_USE_ACCEL PROPERTY STRINGS ${SUPPORTED_ACCELEARTION_TARGETS})

if ((CP2K_USE_ACCEL MATCHES "CUDA") OR (CP2K_USE_ACCEL MATCHES "HIP"))
  set(CP2K_USE_GEMM_OFFLOADING
    OFF
    CACHE BOOL
    "Use SPLA offloading gemm feature to the GPU if it is beneficial. ")
  set(CP2K_USE_GRID_GPU
    ON
    CACHE
    BOOL
    "Enable the hardware accelerated (mostly GPU) backend for grid related functions. It is only effective when general gpu support is enabled."
    )
  set(CP2K_USE_PW_GPU
    ON
    CACHE
    BOOL
    "enable the ffts accelerated (mostly GPU) backend. It is only effective when general gpu support is enabled."
    )
  set(CP2K_USE_DBM_GPU
    ON
    CACHE
    BOOL
    "enable the dbm accelerated (mostly GPU) backend. It is only effective when general gpu support is enabled."
    )

  # set(CP2K_USE_HIP_CUDA
  #   OFF
  #   CACHE BOOL
  #   "Compile hip code for Nvidia gpus. Can be used for debugging hip code when hardware is not available")
endif()

set(CP2K_USE_HIP OFF)
set(CP2K_USE_CUDA OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ${CP2K_SET_POSITION_INDEPENDENT_CODE})

# =================================================================================================
# LANGUAGES AND TESTING

enable_language(CXX)
enable_language(C)
enable_language(Fortran)

find_package(PkgConfig)

# =================================================================================================
# PACKAGE DISCOVERY (compiler configuration can impact package discovery)

find_package(OpenMP REQUIRED COMPONENTS Fortran C CXX)

# =================================== Python this module looks preferably for
# version 3 of Python. If not found, version 2 is searched. In CMake 3.15, if a
# python virtual environment is activated, it will search the virtual
# environment for a python interpreter before searching elsewhere in the system.
# In CMake <3.15, the system is searched before the virtual environment.
if(NOT Python_EXECUTABLE)
  # If the python interpreter isn't specified as a command line option, look for
  # it:
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)
endif()

# get the git hash 
# Get the latest abbreviated commit hash of the working branch
execute_process(
COMMAND git log -1 --format=%h
WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
OUTPUT_VARIABLE CP2K_GIT_HASH
OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
	COMMAND hostnamectl --transient
	WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
	OUTPUT_VARIABLE CP2K_HOST_NAME
	OUTPUT_STRIP_TRAILING_WHITESPACE
	)

add_custom_target(AlwaysCheckGit COMMAND ${CMAKE_COMMAND}
    -DRUN_CHECK_GIT_VERSION=1
    -Dpre_configure_dir=${pre_configure_dir}
    -Dpost_configure_file=${post_configure_dir}
    -DGIT_HASH_CACHE=${GIT_HASH_CACHE}
    -P ${CURRENT_LIST_DIR}/CheckGit.cmake
    BYPRODUCTS ${post_configure_file}
    )

# =================================== MPI

#if (CP2K_USE_MPI)
  get_property(REQUIRED_MPI_COMPONENTS GLOBAL PROPERTY ENABLED_LANGUAGES)
  list(REMOVE_ITEM REQUIRED_MPI_COMPONENTS CUDA) # CUDA does not have an MPI
  # component
  if(NOT CMAKE_CROSSCOMPILING) # when cross compiling, assume the users know
    # what they are doing
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  endif()
  find_package(
    MPI
    COMPONENTS ${REQUIRED_MPI_COMPONENTS}
    REQUIRED)

  if(NOT MPI_Fortran_HAVE_F90_MODULE)
    message(
      FATAL_ERROR
      "\
The listed MPI implementation does not provide the required mpi.mod interface. \
When using the GNU compiler in combination with Intel MPI, please use the \
Intel MPI compiler wrappers. Check the INSTALL.md for more information.")
  endif()
  if("${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v2.1"
      OR "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v3.1")
    message(
      WARNING
      "RMA with ${MPI_Fortran_LIBRARY_VERSION_STRING} is not supported due to issues with its implementation."
      " Please use a newer version of OpenMPI or switch to MPICH if you plan on using MPI-RMA."
      )
  endif()
#endif ()
# =================================== BLAS & LAPACK, PkgConfig
find_package(Lapack REQUIRED) # also calls find_package(BLAS)

# =================================== SMM (Small Matrix-Matrix multiplication)
if(CP2K_USE_LIBXSMM)
  find_package(LibXSMM REQUIRED)
  message(STATUS "-- Using libxsmm for Small Matrix Multiplication")
endif()

# in practice it is always for any decent configuration. But I add a flags to
# turn it off
find_package(DBCSR 2.2.0 REQUIRED)
find_package(SCALAPACK REQUIRED)

# ================================== CUDA / ROCM
# easy for cuda
# a moving target for hip
# ======================================

if ((CP2K_USE_ACCEL MATCHES CUDA) OR (CP2K_USE_ACCEL MATCHES HIP))
  set(GPU_ARCH_NUMBER_K20X 35)
  set(GPU_ARCH_NUMBER_K40 35)
  set(GPU_ARCH_NUMBER_K80 37)
  set(GPU_ARCH_NUMBER_P100 60)
  set(GPU_ARCH_NUMBER_V100 70)
  set(GPU_ARCH_NUMBER_A100 80)
  set(GPU_ARCH_NUMBER_Mi50 gfx906)
  set(GPU_ARCH_NUMBER_Mi100 gfx908)
  set(GPU_ARCH_NUMBER_Mi200 gfx90a)
  set(GPU_ARCH_NUMBER_Mi250X gfx90a)
endif()

if(CP2K_USE_ACCEL MATCHES "CUDA")

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  # P100 is the default target. We need that in cray environment for some weird
  # reason. And it does not harm to define it anyway
  set(CMAKE_CUDA_ARCHITECTURES 60)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  enable_language(CUDA)

  find_package(CUDAToolkit REQUIRED)

  list(FIND SUPPORTED_CUDA_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)

  if(GPU_SUPPORTED EQUAL -1)
    message(
      FATAL_ERROR
      "GPU architecture (${WITH_GPU}) is not supported. Please choose from: ${SUPPORTED_CUDA_ARCHITECTURES}"
      )
  endif()

  # set cuda architecture number and compilation flags
  add_compile_definitions(__OFFLOAD_CUDA
    $<$<COMPILE_LANGUAGE:CUDA>:__OFFLOAD_CUDA>)

  # set cuda architecture number and compilation flags. Taken from dbcsr
  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  set(CMAKE_CUDA_ARCHITECTURES ${ACC_ARCH_NUMBER})
  set(CUDA_ARCHITECTURES ${ACC_ARCH_NUMBER})

  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})
  message(STATUS "GPU profiling enabled: " ${WITH_CUDA_PROFILING})

  if(WITH_CUDA_PROFILING)
    find_library(
      CUDA_NVTOOLSEXT nvToolsExt
      PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
      DOC "Building with CUDA profiling requires the nvToolsExt CUDA library"
      REQUIRED)
    message(STATUS "Found nvToolsExt: ${CUDA_NVTOOLSEXT}")
  endif()

  set(CP2K_USE_CUDA ON)
  message(STATUS "-- CUDA compiler and libraries found")
elseif(CP2K_USE_ACCEL MATCHES "HIP")
  enable_language(HIP)
  # Find hip
  find_package(hip REQUIRED IMPORTED CONFIG)
  find_package(hipfft REQUIRED IMPORTED CONFIG)
  find_package(hipblas REQUIRED IMPORTED CONFIG)

  add_compile_definitions(__OFFLOAD_HIP  __HIP_PLATFORM_AMD__)

  set(CMAKE_HIP_ARCHITECTURES gfx801 gfx900 gfx90a)
  if(NOT CMAKE_BUILD_TYPE)
    set(HIP_RELEASE_OPTIONS "-O3 -g -DNDEBUG")
  elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    set(HIP_RELEASE_OPTIONS "-O3 -g -DNDEBUG")
  elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(HIP_RELEASE_OPTIONS "-O3 -DNDEBUG")
  elseif(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(HIP_RELEASE_OPTIONS "-O0 -g")
  endif()

  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  set(CP2K_USE_HIP ON)

  # if(CP2K_USE_HIP_CUDA)
  #   message(
  #     STATUS
  #     "hip is targeting cuda hardware. We need to check for the presence of cuda"
  #     )
  #   find_package(CUDA 9.0 REQUIRED)
  #   find_package(CUDAToolkit 9.0 REQUIRED)
  #   if(NOT HIP_HCC_FLAGS)
  #     set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} "-arch sm_${GPU_ACC_ARCH_NUMBER}")
  #   endif()
  # else()
  #   if(NOT HIP_HCC_FLAGS)
  #     message(
  #       STATUS
  #       "Using default AMD gpu targets: gfx803, gfx900, gfx906. Set HIP_HCC_FLAGS to override."
  #       )
  #     set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS} --amdgpu-target=${WITH_GPU})
  #   endif()
  # endif()
  # we need hipfft and hipblas
endif()

# ==================================
if(CP2K_USE_ELPA)
  find_package(Elpa REQUIRED)
endif()

if(CP2K_USE_LIBXC)
  find_package(LibXC 5.2.0 REQUIRED)
endif()

if(CP2K_USE_COSMA)
  find_package(cosma REQUIRED)
  get_target_property(COSMA_INCLUDE_DIRS cosma::cosma
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT COSMA_INCLUDE_DIRS)
    set(COSMA_INCLUDE_DIRS "/usr/include;/usr/include/cosma")
  endif()

  # check that cosma::cosma_pxgemm and cosma::cosma_prefixed_pxgemm exist
  if(NOT TARGET cosma::cosma_pxgemm OR NOT TARGET cosma::cosma_prefixed_pxgemm)
    message(ERROR " COSMA needs to be build with scalapack offloading support")
    message(
      ERRRO
      " COSTA_SCALAPACK and COSMA_SCALAPACK should probably be set properly")
  endif()
endif()

if(CP2K_USE_VORI)
  find_package(LibVORI REQUIRED)
endif()

# =================================== FFTW3
# ===========================================

if(CP2K_USE_FFTW3)
  find_package(Fftw REQUIRED)
endif()

# ==================================== QUIP
# ===========================================
if(CP2K_USE_QUIP)
  find_package(Quip REQUIRED)
endif()

# ==================================== libint
# =========================================

if(CP2K_USE_LIBINT2)
  find_package(Libint2 REQUIRED)
endif()

# ==================================== spglib
# =========================================

if(CP2K_USE_SPGLIB)
  find_package(LibSPG REQUIRED)
endif()

if(CP2K_USE_SPLA)
  find_package(SPLA REQUIRED)
  get_target_property(SPLA_INCLUDE_DIRS SPLA::spla
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT SPLA_INCLUDE_DIRS)
    set(SPLA_INCLUDE_DIRS "/usr/include;/usr/include/spla")
  endif()
endif()
# =================================== SIRIUS
# ==========================================

if(CP2K_USE_SIRIUS)
  find_package(sirius REQUIRED)
  get_target_property(SIRIUS_INCLUDE_DIRS sirius::sirius
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT SIRIUS_INCLUDE_DIRS)
    set(SIRIUS_INCLUDE_DIRS "/usr/include;/usr/include/sirius")
  endif()
endif()

if(CP2K_USE_SUPERLU)
  find_package(SuperLU REQUIRED)
endif()

if(CP2K_USE_PARMETIS)
  find_package(Metis)
endif()

if(CP2K_USE_PTSCOTCH)
  find_package(Ptscotch REQUIRED)
endif()

if(CP2K_USE_PEXSI)
  # PEXSI 1.2 uses cmake as build system
  find_package(PEXSI REQUIRED)
endif()

if(CP2K_USE_PLUMED)
  find_package(Plumed REQUIRED)
endif()

# =================================================================================================
# OPTION HANDLING

# make sure that the default build type is RELEASE
set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
    "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
    "${default_build_type}"
    CACHE STRING
    "Choose the type of build, options are: Debug Release Coverage."
    FORCE)
  # set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "Coverage")
endif()

# compiler configuration could have impacted package discovery (above)
include(CompilerConfiguration)
include(CheckCompilerSupport)

# subdirectories
add_subdirectory(src)

include(CustomTargets)

#string(TIMESTAMP CP2K_TIMESTAMP "%Y-%m-%d %H:%M:%S")

message(STATUS "    ")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "-                                                                  -")
message(STATUS "-               Summary of enabled dependencies                    -")
message(STATUS "-                                                                  -")
message(STATUS "--------------------------------------------------------------------")

message(STATUS "    ")
message(STATUS "- BLAS AND LAPACK")
message(STATUS "    ")

message(STATUS "   - vendor :              " ${BLAS_VENDOR})
message(STATUS "   - include directories : " ${BLAS_INCLUDE_DIRS}
  ${LAPACK_INCLUDE_DIR})
message(STATUS "   - libraries :           " ${BLAS_LIBRARIES} " " ${LAPACK_LIBRARIES})

if(CP2K_USE_MPI)
  message(STATUS " ")
  message(STATUS "- MPI")
  message(STATUS "   - include directories : " ${MPI_INCLUDE_DIRS})
  message(STATUS "   - libraries :           " ${MPI_LIBRARIES})
endif()

if(CP2K_USE_SCALAPACK)
  message(STATUS " ")
  message(STATUS "- SCALAPACK:")
  message(STATUS "     - libraries : " ${SCALAPACK_LIBRARIES})
endif()

message(STATUS " - Hardware Acceleration:")
message(STATUS "                         ")
if(CP2K_USE_ACCEL MATCHES "CUDA")
  message(STATUS " ")
  message(STATUS "- CUDA:")
  message(STATUS "     - GPU target architecture : " ${WITH_GPU})
  message(STATUS "     - GPU architecture number : " ${ACC_ARCH_NUMBER})
  message(STATUS "     - GPU profiling enabled :   " ${WITH_CUDA_PROFILING})
endif()

if(CP2K_USE_ACCEL MATCHES "HIP")
  message(STATUS " ")
  message(STATUS "- HIP:")
  message(STATUS "    - GPU target architecture : " ${WITH_GPU})
  message(STATUS "    - GPU architecture number : " ${ACC_ARCH_NUMBER})
  # message(STATUS "GPU profiling enabled: " ${WITH_CUDA_PROFILING})
endif()

if((CP2K_USE_ACCEL MATCHES "CUDA") OR (CP2K_USE_ACCEL MATCHES "HIP"))
  message(STATUS " ")
  message(STATUS
    " Note : Enabling hardware acceleration enable acceleration of the grid, pw, and dbm modules by default"
    )
  message(STATUS "    - GRID module : " ${CP2K_USE_GRID_GPU})
  message(STATUS "    - PW   module : " ${CP2K_USE_PW_GPU})
  message(STATUS "    - DBM  module : " ${CP2K_USE_DBM_GPU})
endif()

if(CP2K_USE_LIBXC)
  message(STATUS " ")
  message(STATUS
    "- LIBXC (note to package managers : libxc can be build with cmake as well)"
    )
  message(STATUS "   - include directories : " ${LIBXC_INCLUDE_DIRS})
  message(STATUS "   - libraries :           " ${LIBXC_LIBRARIES})
endif()

if(CP2K_USE_FFTW3)
  message(STATUS " ")
  message(STATUS "- FFTW3")
  message(STATUS "   - include directories : " ${FFTW3_INCLUDE_DIRS})
  message(STATUS "   - libraries :           " ${FFTW3_LIBRARIES})
endif()

if(CP2K_USE_LIBXSMM)
  message(STATUS " ")
  message(STATUS "- libxsmm")
  message(STATUS "   - include directories : " ${LIBXSMM_INCLUDE_DIRS})
  message(STATUS "   - libraries :           " ${LIBXSMM_LIBRARIES})
endif()

if(CP2K_USE_SPLA)
  message(STATUS " ")
  message(STATUS "- SPLA :")
  message(STATUS "   - lbraries : " "${SPLA_INCLUDE_DIRS}")
endif()

if(CP2K_USE_SIRIUS)
  message(STATUS " ")
  message(STATUS "- SIRIUS :")
  message(STATUS "   - include directories : " ${SIRIUS_INCLUDE_DIRS})
  message(STATUS "   - libraries           : " ${SIRIUS_LIBRARIES})

  message(STATUS "   - dependencies : ")
  message(STATUS "       - spla")
  message(STATUS "       - SpFFT")
  message(STATUS "       - SPGLIB")
  message(STATUS "       - LibXC")
  message(STATUS "       - fftw3")
  message(STATUS "       - hdf5")
  message(STATUS "       - GSL")
  if(CP2K_USE_VDWXC)
    message(STATUS "     - VDWXC")
  endif()
endif()

if(CP2K_USE_COSMA)
  message(STATUS " ")
  message(STATUS "- COSMA")
  message(STATUS "   - include directories : " ${COSMA_INCLUDE_DIRS})
  # message(STATUS "  - libraries           : " ${COSMA_LIBRARIES})
endif()

if(CP2K_USE_QUIP)
  message(STATUS " ")
  message(STATUS "- QUIP")
endif()

if(CP2K_USE_PEXSI)
  message(STATUS " ")
  message(STATUS "- PEXSI")
endif()

if(CP2K_USE_LIBINT2)
  message(STATUS " ")
  message(STATUS "- libint2")
  message(STATUS "   - include directories : " ${LIBINT2_INCLUDE_DIRS})
  message(STATUS "   - libraries :           " ${LIBINT2_LIBRARIES})
endif()

if(CP2K_USE_VORI)
  message(STATUS " ")
  message(STATUS "- libvori")
  message(STATUS "   - include directories : " ${LIBVORI_INCLUDE_DIRS})
  message(STATUS "   - libraries :           " ${LIBVORI_LIBRARIES})
endif()

if(CP2K_USE_SPGLIB)
  message(STATUS " ")
  message(STATUS "- spglib")
  message(STATUS "   - include directories : " ${LIBSPG_INCLUDE_DIRS})
  message(STATUS "   - libraries           : " ${LIBSPG_LIBRARIES})
endif()

message(STATUS "    ")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "-                                                                  -")
message(STATUS "-        List of dependencies not included in this build           -")
message(STATUS "-                                                                  -")
message(STATUS "--------------------------------------------------------------------")

message(STATUS " List of dependencies not included in this build :")

if (NOT CP2K_USE_SPGLIB)
  message(STATUS "   - SPGLIB")
endif ()

if (NOT CP2K_USE_MPI)
  message(STATUS "   - MPI")
endif()

if (NOT CP2K_USE_SCALAPACK)
  message(STATUS "   - scalapack")
endif()

if (NOT CP2K_USE_COSMA)
  message(STATUS "   - COSMA")
endif()

if (NOT CP2K_USE_SPLA)
  message(STATUS "   - SPLA")
endif()

if (${CP2K_USE_ACCEL} MATCHES "NONE")
  message(STATUS "   - GPU acceleration is off")
endif()

if (NOT CP2K_USE_ELPA)
  message(STATUS "   - ELPA")
endif()

if (NOT CP2K_USE_PLUMMED)
  message(STATUS "   - PLUMMED")
endif()

if (NOT CP2K_USE_QUIP)
  message(STATUS "   - QUIP")
endif()

if (NOT CP2K_USE_LIBXSMM)
  message(STATUS "   - LIBXSMM")
endif()

if (NOT CP2K_USE_LIBINT2)
  message(STATUS "   - LIBINT2")
endif()

if (NOT CP2K_USE_LIBXC)
  message(STATUS "   - LIBXC")
endif()

if (NOT CP2K_USE_VORI)
  message(STATUS "   - LIBVORI")
endif()

if (NOT CP2K_USE_FFTW3)
  message(STATUS "   - FFTW3")
endif()

if (NOT CP2K_USE_PEXSI)
  message(STATUS "   - PEXSI")
endif()

message(STATUS "                                                                    ")
message(STATUS "                                                                    ")
