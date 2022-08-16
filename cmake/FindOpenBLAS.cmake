# Copyright (c) 2019 ETH Zurich
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# .rst: FindOPENBLAS
# -----------
#
# This module tries to find the OPENBLAS library.
#
# The following variables are set
#
# ::
#
# OPENBLAS_FOUND           - True if openblas is found OPENBLAS_LIBRARIES - The
# required libraries OPENBLAS_INCLUDE_DIRS    - The required include directory
#
# The following import target is created
#
# ::
#
# OpenBLAS::openblas

# set paths to look for library from ROOT variables.If new policy is set,
# find_library() automatically uses them.
if(NOT POLICY CMP0074)
  list(APPEND _OPENBLAS_PATHS ${OPENBLAS_ROOT} $ENV{OPENBLAS_ROOT})
endif()
find_package(PkgConfig)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(OPENBLAS openblas)
endif()

# try the openblas module of openblas library Maybe we are lucky it is installed
# find_package(OPENBLAS QUIET)

if(OPENBLAS_FOUND)
  message("Found openblas")
endif()

if(NOT OPENBLAS_FOUND)
  find_library(
    OPENBLAS_LIBRARIES_32
    NAMES "openblas"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas" "openblas/lib" "openblas/lib64" "openblas")
  find_library(
    OPENBLAS_LIBRARIES_32_PTHREADS
    NAMES "openblas"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas-pthreads" "openblas/lib" "openblas/lib64"
                  "openblas")
  find_library(
    OPENBLAS_LIBRARIES_32_OPENMP
    NAMES "openblas"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas-openmp" "openblas/lib" "openblas/lib64" "openblas")
  find_library(
    OPENBLAS_LIBRARIES_64
    NAMES "openblas64"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas64" "openblas/lib" "openblas/lib64" "openblas")
  find_library(
    OPENBLAS_LIBRARIES_PTHREADS_64
    NAMES "openblas64p"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas64-pthreads" "openblas/lib" "openblas/lib64"
                  "openblas")
  find_library(
    OPENBLAS_LIBRARIES_OPENMP_64
    NAMES "openblas64p"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas64-openmp" "openblas/lib" "openblas/lib64"
                  "openblas")
  find_path(
    OPENBLAS_INCLUDE_DIRS
    NAMES "cblas.h"
    HINTS ${_OPENBLAS_PATHS}
    PATH_SUFFIXES "openblas" "openblas/include" "include/openblas")
  set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARIES_32})
endif()

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS REQUIRED_VARS OPENBLAS_INCLUDE_DIRS
                                                         OPENBLAS_LIBRARIES)

# add target to link against
if(OPENBLAS_FOUND)
  if(NOT TARGET OpenBLAS::openblas)
    add_library(OpenBLAS::openblas INTERFACE IMPORTED)
    add_library(OpenBLAS::blas INTERFACE IMPORTED)
  endif()
  set_property(TARGET OpenBLAS::openblas PROPERTY INTERFACE_LINK_LIBRARIES
                                                  ${OPENBLAS_LIBRARIES})
  set_property(TARGET OpenBLAS::openblas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                  ${OPENBLAS_INCLUDE_DIRS})
  set_property(TARGET OpenBLAS::blas PROPERTY INTERFACE_LINK_LIBRARIES
                                              ${OPENBLAS_LIBRARIES})
  set_property(TARGET OpenBLAS::blas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                              ${OPENBLAS_INCLUDE_DIRS})
  set(BLAS_VENDOR "OpenBLAS")
endif()

# prevent clutter in cache
mark_as_advanced(BLAS_VENDOR OPENBLAS_FOUND OPENBLAS_LIBRARIES
                 OPENBLAS_INCLUDE_DIRS)
