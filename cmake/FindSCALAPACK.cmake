# Copyright (c) 2019 ETH Zurich, Simon Frasch
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

# .rst: FindSCALAPACK
# -----------
#
# This module searches for the ScaLAPACK library.
#
# To specify a vendor use SCALAPACK_BACKENDS. Two values are possible "generic,
# mkl, sci"
#
# The following variables are set
#
# ::
#
# SCALAPACK_FOUND           - True if double precision ScaLAPACK library is
# found SCALAPACK_FLOAT_FOUND     - True if single precision ScaLAPACK library
# is found SCALAPACK_LIBRARIES       - The required libraries
#
# The following import target is created
#
# ::
#
# SCALAPACK::SCALAPACK

# set paths to look for library

set(SCALAPACK_FOUND FALSE)
set(SCALAPACK_LIBRARIES)

# check if we have mkl as blas library or not and pick the scalapack from mkl
# distro if found

if(SCALAPACK_VENDOR STREQUAL "GENERIC")
  if(TARGET MKL::scalapack_link)
    message("-----------------------------------------------------------------")
    message("-                  FindScalapack warning                        -")
    message("-----------------------------------------------------------------")
    message("                                                                 ")
    message(
      WARNING
        "You may want to use mkl implementation of scalapack. To do this add -DSCALAPACK_VENDOR=MKL to the cmake command line"
    )
  endif()

  if(TARGET SCI::scalapack_link)
    message("-----------------------------------------------------------------")
    message("-                  FindScalapack warning                        -")
    message("-----------------------------------------------------------------")
    message("                                                                 ")
    message(
      WARNING
        "You may want to use Cray implementation of scalapack. To do this add -DSCALAPACK_VENDOR=SCI to the cmake command line"
    )
    message("                                                                 ")
    message("                                                                 ")
  endif()

  set(_SCALAPACK_PATHS ${SCALAPACK_ROOT} $ENV{SCALAPACK_ROOT})
  set(_SCALAPACK_INCLUDE_PATHS)

  set(_SCALAPACK_DEFAULT_PATH_SWITCH)

  if(_SCALAPACK_PATHS)
    # disable default paths if ROOT is set
    set(_SCALAPACK_DEFAULT_PATH_SWITCH NO_DEFAULT_PATH)
  else()
    # try to detect location with pkgconfig
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(SCALAPACK "scalapack")
    endif()
  endif()

  # this should be enough for detecting scalapack compiled by hand. If scalapack
  # is vendor specific then we sahould have a target blas::scalapack available.
  # it removes the problem of modifying too many files when we add a vendor
  # specific blas/lapack/scalapack implementation

  if(NOT SCALAPACK_FOUND)
    find_library(
      SCALAPACK_LIBRARIES
      NAMES "scalapack" "scalapack-mpich" "scalapack-openmpi"
      HINTS ${_SCALAPACK_PATHS}
      PATH_SUFFIXES "lib" "lib64" "lib64/openmpi/lib" "lib64/openmpi/lib64"
                    "lib/openmpi/lib" ${_SCALAPACK_DEFAULT_PATH_SWITCH})
  endif()
elseif(TARGET MKL::scalapack_link)
  # we have mkl check for the different mkl target
  get_target_property(SCALAPACK_LIBRARIES MKL::scalapack_link
                      INTERFACE_LINK_LIBRARIES)

  set(SCALAPACK_FOUND yes)
elseif(TARGET SCI::scalapack_link)
  # we have mkl check for the different mkl target
  get_target_property(SCALAPACK_LIBRARIES SCI::scalapack_link
                      INTERFACE_LINK_LIBRARIES)

  set(SCALAPACK_FOUND yes)
endif()

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCALAPACK REQUIRED_VARS SCALAPACK_LIBRARIES)
# prevent clutter in cache

# add target to link against
if(SCALAPACK_FOUND)
  if(NOT TARGET SCALAPACK::scalapack)
    add_library(SCALAPACK::scalapack INTERFACE IMPORTED)
    set_property(TARGET SCALAPACK::scalapack PROPERTY INTERFACE_LINK_LIBRARIES
                                                      ${SCALAPACK_LIBRARIES})
  endif()
endif()
mark_as_advanced(SCALAPACK_FOUND SCALAPACK_LIBRARIES)
