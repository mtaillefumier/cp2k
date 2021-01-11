cmake_minimum_required(VERSION 3.17)

# include our cmake snippets
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# =================================================================================================
# REQUIRE OUT-OF-SOURCE BUILDS
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if (EXISTS "${LOC_PATH}")
  message(
    FATAL_ERROR
      "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory."
  )
endif ()

# =================================================================================================
# PROJECT AND VERSION
include(GetGitRevisionDescription)

git_describe(GIT_DESC)

if (GIT_DESC)
  string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR
                       "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" VERSION_PATCH
                       "${GIT_DESC}")
  string(REGEX REPLACE "^v[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" VERSION_GIT
                       "${GIT_DESC}")

  git_local_changes(GIT_STATE)
  if ("${GIT_STATE}" STREQUAL "DIRTY")
    set(VERSION_GIT "${VERSION_GIT}-dirty")
  endif ()

  execute_process(
    COMMAND git log -1 --format=%ai
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  # take only the date from the git timestamp:
  string(REGEX REPLACE "^([0-9\\-]+) .*" "\\1" VERSION_DATE
                       "${GIT_COMMIT_DATE}")
else ()
  file(STRINGS VERSION VERSION_INFO)
  foreach (line ${VERSION_INFO})
    if (${line} MATCHES "^([^#].*)=[ \t]*(.*)$")
      set(key ${CMAKE_MATCH_1})
      set(value ${CMAKE_MATCH_2})
      string(REGEX REPLACE "[ \t\n]+$" "" key "${key}")
      string(REGEX REPLACE "[ \t\n]+$" "" value "${value}")
      set(VERSION_${key} "${value}")
      continue()
    endif ()
  endforeach ()
endif ()

project(
  cp2k
  DESCRIPTION
  "CP2K"
  LANGUAGES Fortran C
)
set(cp2k_VERSION
    ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_GIT})
set(cp2k_APIVERSION ${VERSION_MAJOR}.${VERSION_MINOR})

# =================================================================================================
# OPTIONS
include(CMakeDependentOption)

option(USE_MPI "Build with MPI support" ON)

set(USE_SMM
    "blas"
    CACHE STRING
          "Small Matrix Multiplication implementation to use (default: blas)")
set_property(CACHE USE_SMM PROPERTY STRINGS blas libxsmm)

option(USE_SIRIUS "Enable plane wave dft calculations with sirius" NO)
option(USE_FFTW3 "Use fftw3 for the calculating fast fourier transforms" NO)
option(USE_ELPA "Enable elpa support" NO)
option(USE_SCALPACK "Enable scalpack support." NO)
option(USE_PTSCOTCH "Enable ptscotch support" NO)
option(USE_CUDA "Enable cuda support" NO)
option(USE_HIP "Enable AMD GPU support" NO)
option(USE_COSMA "COSMA is a drop in replacement of scalapack dgemm" NO)
option(USE_LIBINT2 "Enable libint2 support" NO)
option(USE_VORI "Enable libvori support" NO)
option(USE_PEXSI "Enable pexsi support" NO)
option(USE_QUIP "Enable quip support" NO)
option(USE_SPGLIB "Enable spglib support" NO)
option(USE_LIBXC "Enable libxc support" NO)
option(USE_SCALAPACK "Enable scalapack support" NO)

set(SUPPORTED_CUDA_ARCHITECTURES K20X K40 K80 P100 V100)
set(SUPPORTED_HIP_ARCHITECTURES Mi50)
set(WITH_GPU
  "P100"
  CACHE STRING
  "Set the CUDA GPU architecture if HIP is enabled (default: P100)")
set_property(CACHE WITH_GPU PROPERTY STRINGS ${SUPPORTED_CUDA_ARCHITECTURES}
  ${SUPPORTED_HIP_ARCHITECTURES})

# =================================================================================================
# LANGUAGES AND TESTING

if (WITH_C_API AND WITH_EXAMPLES OR USE_SIRIUS OR USE_ROCM OR USE_CUDA)
  enable_language(CXX)
endif ()

# =================================================================================================
# PACKAGE DISCOVERY (compiler configuration can impact package discovery)

find_package(OpenMP REQUIRED COMPONENTS Fortran C)

find_package(DBCSR 2.1.0 CONFIG REQUIRED)

# =================================== BLAS & LAPACK, PkgConfig
find_package(PkgConfig)
find_package(LAPACK REQUIRED) # also calls find_package(BLAS)

# =================================== Python this module looks preferably for
# version 3 of Python. If not found, version 2 is searched. In CMake 3.15, if a
# python virtual environment is activated, it will search the virtual
# environment for a python interpreter before searching elsewhere in the system.
# In CMake <3.15, the system is searched before the virtual environment.
if (NOT Python_EXECUTABLE)
  # If the python interpreter isn't specified as a command line option, look for
  # it:
  find_package(
    Python
    COMPONENTS Interpreter
    REQUIRED)
endif ()

# =================================== MPI
if (USE_MPI)
  get_property(REQUIRED_MPI_COMPONENTS GLOBAL PROPERTY ENABLED_LANGUAGES)
  list(REMOVE_ITEM REQUIRED_MPI_COMPONENTS CUDA) # CUDA does not have an MPI
                                                 # component
  if (NOT CMAKE_CROSSCOMPILING) # when cross compiling, assume the users know
                                # what they are doing
    set(MPI_DETERMINE_LIBRARY_VERSION TRUE)
  endif ()
  find_package(
    MPI
    COMPONENTS ${REQUIRED_MPI_COMPONENTS}
    REQUIRED)

  if (NOT MPI_Fortran_HAVE_F90_MODULE)
    message(
      FATAL_ERROR
        "\
The listed MPI implementation does not provide the required mpi.mod interface. \
When using the GNU compiler in combination with Intel MPI, please use the \
Intel MPI compiler wrappers. Check the INSTALL.md for more information.")
  endif ()
  if ("${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v2.1"
      OR "${MPI_Fortran_LIBRARY_VERSION_STRING}" MATCHES "Open MPI v3.1")
    message(
      WARNING
        "RMA with ${MPI_Fortran_LIBRARY_VERSION_STRING} is not supported due to issues with its implementation."
        " Please use a newer version of OpenMPI or switch to MPICH if you plan on using MPI-RMA."
    )
  endif ()
endif ()

# =================================== SMM (Small Matrix-Matrix multiplication)
if (USE_SMM MATCHES "blas")
  message("-- Using BLAS for Small Matrix Multiplication")
elseif (USE_SMM MATCHES "libxsmm")
  # rely on pkg-config in order to link against libxsmm
  pkg_check_modules(deps REQUIRED IMPORTED_TARGET GLOBAL libxsmmf)
  message("-- Using libxsmm for Small Matrix Multiplication")
else ()
  message(FATAL_ERROR "Unknown SMM library specified")
endif ()

# ================================== CUDA / ROCM ======================================

if (USE_CUDA AND USE_HIP)
  message(error "CUDA (for nvidia GPU) and ROCM/HIP programming models are mutually exclusive")
endif ()

if (USE_CUDA OR USE_HIP)
  set(GPU_ARCH_NUMBER_K20X 35)
  set(GPU_ARCH_NUMBER_K40 35)
  set(GPU_ARCH_NUMBER_K80 37)
  set(GPU_ARCH_NUMBER_P100 60)
  set(GPU_ARCH_NUMBER_V100 70)
  set(GPU_ARCH_NUMBER_Mi50 gfx906)
endif ()

if (USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 9.0 REQUIRED QUIET)

  list(FIND SUPPORTED_CUDA_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)
  if (GPU_SUPPORTED EQUAL -1)
    message(
      FATAL_ERROR
        "GPU architecture requested (${WITH_GPU}) is not supported. Please choose from: ${SUPPORTED_CUDA_ARCHITECTURES}"
    )
  endif ()

  # assume that the backend compiler for nvcc understands the -std=c++11
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # set cuda architecture number and compilation flags
  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  # TODO: use CMAKE_CUDA_RUNTIME_LIBRARY with CMake 3.17+ and CUDA_ARCHITECTURES
  # with CMake 3.18+
  string(APPEND CMAKE_CUDA_FLAGS " -arch=sm_${ACC_ARCH_NUMBER} --cudart static")
  add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:__CUDA>)
  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})
  message(STATUS "GPU profiling enabled: " ${WITH_CUDA_PROFILING})

  # =================================== BLAS on GPU backend
  find_library(CUBLAS cublas HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
  if (NOT CUBLAS)
    message(
      FATAL_ERROR
        "cuBLAS library not found but support required for DBCSR's CUDA backend"
    )
  else ()
    message(STATUS "Found cuBLAS: ${CUBLAS}")
  endif ()

  if (WITH_CUDA_PROFILING)
    find_library(
      CUDA_NVTOOLSEXT nvToolsExt
      PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
      DOC "Building with CUDA profiling requires the nvToolsExt CUDA library"
          REQUIRED)
    message(STATUS "Found nvToolsExt: ${CUDA_NVTOOLSEXT}")
  endif ()

  message("-- CUDA compiler and libraries found")
endif ()

if (USE_HIP)
  # Make sure the GPU required is supported
  list(FIND SUPPORTED_HIP_ARCHITECTURES ${WITH_GPU} GPU_SUPPORTED)
  if (GPU_SUPPORTED EQUAL -1)
    message(
      FATAL_ERROR
        "GPU architecture requested (${WITH_GPU}) is not supported. Please choose from: ${SUPPORTED_HIP_ARCHITECTURES}"
    )
  endif ()

  # Set path to HIP installation, include HIP cmake utilities
  if (NOT DEFINED HIP_PATH)
    if (NOT DEFINED ENV{HIP_PATH})
      set(HIP_PATH
          "/opt/rocm/hip"
          CACHE PATH "Path to HIP installation")
    else ()
      set(HIP_PATH
          $ENV{HIP_PATH}
          CACHE PATH "Path to HIP installation")
    endif ()
  endif ()
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${HIP_PATH}/cmake")

  if (NOT DEFINED ROCM_PATH)
    if (NOT DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH
          "/opt/rocm"
          CACHE PATH "Path to ROCm installation")
    else ()
      set(ROCM_PATH
          $ENV{ROCM_PATH}
          CACHE PATH "Path to ROCm installation")
    endif ()
  endif ()

  # Find HIP package
  find_package(HIP)
  if (HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
  else ()
    message(
      FATAL_ERROR
        "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location."
    )
  endif ()

  # Find hiprtc library (adds support for JIT-ing in HIP)
  find_library(ROCM_HIPRTC_LIB amdhip64 HINTS ${HIP_PATH}/lib)
  if (NOT ROCM_HIPRTC_LIB)
    message(
      FATAL_ERROR "HIPRTC (HIP library for just-in-time compilation) not found")
  endif ()
  set(ENV{HIP_PATH} /opt/rocm/hip) # workaround bug in hiprtc.cpp

  # Set platform to compile for (NVIDIA-nvcc or ROCm-hcc) as well as
  # corresponding architecture and flags adapted from:
  # https://github.com/ROCmSoftwarePlatform/hipDNN/blob/master/CMakeLists.txt
  execute_process(COMMAND ${HIP_PATH}/bin/hipconfig -P
                  OUTPUT_VARIABLE HIP_PLATFORM)
  message(STATUS "Compiling for platform: " ${HIP_PLATFORM})

  # set appropriate compilation flags depending on platform
  set(ACC_ARCH_NUMBER ${GPU_ARCH_NUMBER_${WITH_GPU}})
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -D__HIP -O3")
  if (${HIP_PLATFORM} STREQUAL "nvcc")
    set(HIP_HIPCC_FLAGS
        "${HIP_HIPCC_FLAGS} -std=c++11 -arch=sm_${ACC_ARCH_NUMBER} --cudart static"
    )
  else ()
    set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -fPIC")
  endif ()
  message(STATUS "GPU target architecture: " ${WITH_GPU})
  message(STATUS "GPU architecture number: " ${ACC_ARCH_NUMBER})
  message(STATUS "HIPCC flags: " ${HIP_HIPCC_FLAGS})
  if (USE_OPENMP)
    set(HIP_OpenMP_FLAGS "-L${ROCM_PATH}/llvm/lib -lomp")
    message(STATUS "HIP OpenMP linking flags: ${HIP_OpenMP_FLAGS}")
  endif ()

  # =================================== BLAS on GPU backend
  find_library(HIPBLAS hipblas HINTS ${HIP_PATH}/../lib) # /opt/rocm/lib
  if (NOT HIPBLAS)
    message(
      FATAL_ERROR
        "hipBLAS library not found but support required for DBCSR's HIP backend"
    )
  else ()
    message(STATUS "Found hipBLAS: ${HIPBLAS}")
  endif ()
endif()

# ==================================
if (USE_ELPA)
  find_package(elpa REQUIRED)
endif ()

if (USE_LIBXC)
  find_package(LibXC 4.3.4 REQUIRED)
  message("-- Found libxc")
endif ()

if (USE_COSMA)
  find_package(Cosma REQUIRED)
endif ()

if (USE_VORI)
  find_package(LibVORI REQUIRED)
endif ()

# =================================== FFTW3 ===========================================

if (USE_FFTW3)
  find_package(FFTW)
endif ()

# ==================================== scalapack ======================================
if (USE_SCALAPACK)
  find_package(scalapack REQUIRED)
endif ()

# ==================================== QUIP
if (USE_QUIP)
  find_package(quip REQUIRED)
endif()

# ==================================== libint

if (USE_LIBINT2)
  find_package(LibINT2 REQUIRED)
endif ()

# ==================================== spglib

if (USE_SPGLIB)
  find_package(LibSPG REQUIRED)
endif ()



# =================================== SIRIUS ==========================================

if (USE_SIRIUS)
  find_package(SPLA REQUIRED)
  find_package(SpFFT REQUIRED)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  find_package(GSL REQUIRED)

  if (NOT USE_SPGLIB)
    find_package(LibSPG)
  endif ()

  if (NOT USE_LIBXC)
    find_package(LibXC 4.3.4 REQUIRED)
  endif ()

  if (NOT USE_FFTW3)
    find_package(FFTW REQUIRED)
  endif ()

  find_package(LibVDWXC)

  find_package(sirius REQUIRED)
  message("-- SIRIUS library was found")
endif()

if (USE_PEXSI)
  find_package(ParMETIS required)
  find_package(Superlu required)
  find_package(pexsi required)
endif()


# =================================================================================================
# OPTION HANDLING

# make sure that the default build type is RELEASE
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
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
endif ()

# compiler configuration could have impacted package discovery (above)
include(CompilerConfiguration)
include(CheckCompilerSupport)

# subdirectories
add_subdirectory(src)

include(CustomTargets)

message("    ")
message("--------------------------------------------------------------------")
message("-                                                                  -")
message("-               Summary of enabled dependencies                    -")
message("-                                                                  -")
message("--------------------------------------------------------------------")


if (USE_CUDA)
  message(" ")
  message("  CUDA:")
  message("     GPU target architecture: " ${WITH_GPU})
  message("     GPU architecture number: " ${ACC_ARCH_NUMBER})
  message("     GPU profiling enabled: " ${WITH_CUDA_PROFILING})
endif ()

if (USE_HIP)
  message(" ")
  message("  HIP:")
  message("     GPU target architecture: " ${WITH_GPU})
  message("     GPU architecture number: " ${ACC_ARCH_NUMBER})
#  message(STATUS "GPU profiling enabled: " ${WITH_CUDA_PROFILING})
endif ()

if (USE_LIBXC)
  message(" ")
  message("  LIBXC")
endif ()

if (USE_FFTW)
  message(" ")
  message("  FFTW3")
endif ()

if (USE_LIBXSMM)
  message(" ")
  message("  libxsmm")
endif ()

if (USE_SIRIUS)
  message(" ")
  message("  SIRIUS :")
  message("       - SPLA")
  message("       - SpFFT")
  message("       - SPGLIB")
  message("       - LibXC")
  message("       - fftw3")
  message("       - hdf5")
  message("       - GSL")
  if (USE_VDWXC)
    message("       - VDWXC")
  endif ()
endif ()

if (USE_COSMA)
  message(" ")
  message("  COSMA")
endif ()

if (USE_QUIP)
  message(" ")
  message("  QUIP")
endif ()

if (USE_PEXSI)
  message(" ")
  message("  PEXSI")
endif ()

if (USE_MPI)
  message(" ")
  message("  MPI")
endif ()

if (USE_LIBINT2)
  message(" ")
  message("  libint2")
endif ()

message("                                                                    ")
message("                                                                    ")
