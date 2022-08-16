# Copyright (c) 2022- ETH Zurich
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

# .rst: FindARMPL
# -----------
#
# This module search the flexiblas library.
#
# The following variables are set
#
# ::
#
# FLEXIBLAS_FOUND
# FLEXIBLAS_LIBRARIES
# FLEXIBLAS_INCLUDE_DIRS
#
# The following import targets are created
#
# ::
#
# FLEXIBLAS::flexiblas
# FLEXIBLAS::blas

include(FindPackageHandleStandardArgs)

# try first with pkg-config
find_package(PkgConfig QUIET)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(FLEXIBLAS flexiblas QUIET)
endif()

# manual; search
if(NOT FLEXIBLAS_FOUND)
  find_library(
    FLEXIBLAS_LIBRARIES
    NAMES "flexiblas"
    HINTS ENV
          FLEXIBLAS_ROOT
          ENV
          FLEXIBLASROOT
          ENV
          EB_FLEXIBLAS_ROOT
          ENV
          EBFLEXIBLAS_ROOT)
endif()

# search for include directories anyway
if(NOT FLEXIBLAS_INCLUDE_DIRS)
  find_path(
    FLEXIBLAS_INCLUDE_DIRS
    NAMES "flexiblas.h"
    PATH_SUFFIXES "flexiblas"
    HINTS
    HINTS ENV
          FLEXIBLAS_ROOT
          ENV
          FLEXIBLASROOT
          ENV
          EB_FLEXIBLAS_ROOT
          ENV
          EBFLEXIBLAS_ROOT
    NO_DEFAULT_PATH)
endif()

find_package_handle_standard_args(FlexiBLAS DEFAULT_MSG FLEXIBLAS_INCLUDE_DIRS
                                  FLEXIBLAS_LIBRARIES)

if(NOT FLEXIBLAS_FOUND)
  set(FLEXIBLAS_FOUND ON)
  set(BLAS_VENDOR "FlexiBLAS")
endif()

if(FLEXIBLAS_FOUND AND NOT TARGET FlexiBLAS::flexiblas)
  add_library(FlexiBLAS::flexiblas INTERFACE IMPORTED)
  if(FLEXIBLAS_INCLUDE_DIRS)
    set_target_properties(
      FlexiBLAS::flexiblas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                      "${FLEXIBLAS_INCLUDE_DIRS}")
  endif()
  set_target_properties(
    FlexiBLAS::flexiblas PROPERTIES INTERFACE_LINK_LIBRARIES
                                    "${FLEXIBLAS_LIBRARIES}")
  add_library(FlexiBLAS::blas INTERFACE IMPORTED)
  set_target_properties(FlexiBLAS::blas PROPERTIES INTERFACE_LINK_LIBRARIES
                                                   "${FLEXIBLAS_LIBRARIES}")
endif()

mark_as_advanced(FLEXIBLAS_FOUND FLEXIBLAS_INCLUDE_DIRS
                 FLEXIBLAS_INCLUDE_LIBRARIES)
