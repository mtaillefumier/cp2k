# Copyright (c) 2022 ETH Zurich
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
# This module tries to find the ATLAS library.
#
# The following variables are set
#
# ::
#
# ARMPL_FOUND
# ARMPL_LIBRARIES
# ARMPL_INCLUDE_DIRS
#
# The following import targets are created
#
# ::
#
# ARMPL::armpl ARMPL::blas

include(FindPackageHandleStandardArgs)

# Check for 64bit Integer support
if(BLAS_INTERFACE MATCHES "64bits")
  set(BLAS_armpl_LIB "armpl_ilp64")
else()
  set(BLAS_armpl_LIB "armpl_lp64")
endif()

# Check for OpenMP support, VIA BLAS_VENDOR of Arm_mp or Arm_ipl64_mp
if(BLAS_THREADING MATCHES "openmp")
  set(BLAS_armpl_LIB "${BLAS_armpl_LIB}_mp")
endif()

find_library(ARMPL ARMPL_LIBRARIES NAMES "${BLAS_armpl_LIB}")

find_path(
  ARMPL_INCLUDE_DIRS
  NAMES armpl.h
  PATH_SUFFIXES inc include
  HINTS ENV{ARMPLROOT} ENV{ARMPL_ROOT} ENV{EBARMPL_ROOT} ENV{EBARMPLROOT})

# check if found
find_package_handle_standard_args(Armpl REQUIRED_VARS ARMPL_INCLUDE_DIRS
                                                      ARMPL_LIBRARIES)

# add target to link against
if(ARMPL_FOUND)
  if(NOT TARGET ARMPL::armpl)
    add_library(ARMPL::armpl INTERFACE IMPORTED)
    add_library(ARMPL::blas INTERFACE IMPORTED)
  endif()
  set_property(TARGET ARMPL::armpl PROPERTY INTERFACE_LINK_LIBRARIES
                                            ${ARMPL_LIBRARIES})
  set_property(TARGET ARMPL::armpl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                            ${ARMPL_INCLUDE_DIRS})
  set_property(TARGET ARMPL::blas PROPERTY INTERFACE_LINK_LIBRARIES
                                           ${ARMPL_LIBRARIES})
  set_property(TARGET ARMPL::blas PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                           ${ARMPL_INCLUDE_DIRS})
endif()

mark_as_advanced(ARMPL_FOUND ARMPL_LIBRARIES ARMPL_INCLUDE_DIRS)
