# * Find the FFTW library
#
# Usage: find_package(FFTW [REQUIRED] [QUIET] )
#
# It sets the following variables: FFTW3_FOUND               ... true if fftw is
# found on the system FFTW3_LIBRARIES           ... full path to fftw library
# FFTW3_INCLUDE_DIRS        ... fftw include directory
#
# The following variables will be checked by the function FFTW3_ROOT ... if set,
# the libraries are exclusively searched under this path FFTW3_LIBRARIES ...
# fftw library to use FFTW3_INCLUDE_DIRS       ... fftw include directory
#

include(FindPackageHandleStandardArgs)
# Check if we can use PkgConfig
find_package(PkgConfig)

if(NOT FFTW_ROOT AND ENV{FFTWDIR})
  set(FFTW_ROOT $ENV{FFTWDIR})
endif()

# First try with pkg
if(PKG_CONFIG_FOUND)
  pkg_search_module(FFTW3 fftw3)
endif()

if(NOT FFTW3_FOUND)
  # the fftw3 library comes 4 different flavours, single, double, and long
  # double. Search for all of them. the double precision is almost always
  # installed

  find_library(
    FFTW3_LIB
    NAMES "fftw3"
    PATHS ${FFTW_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    HINTS ENV
          FFTW_DIR
          ENV
          FFTW3_DIR
          ENV
          FFTW_ROOT
          ENV
          FFTW3_ROOT
    NO_DEFAULT_PATH)

  find_library(
    FFTW3_LIB_FLOAT
    NAMES "fftw3f"
    PATH_SUFFIXES "lib" "lib64"
    PATHS ${FFTW_ROOT}
    HINTS ENV
          FFTW_DIR
          ENV
          FFTW3_DIR
          ENV
          FFTW_ROOT
          ENV
          FFTW3_ROOT
    NO_DEFAULT_PATH)

  find_library(
    FFTW3_LIB_LONG_DOUBLE
    NAMES "fftw3l"
    PATH_SUFFIXES "lib" "lib64"
    PATHS ${FFTW_ROOT}
    HINTS ENV
          FFTW_DIR
          ENV
          FFTW3_DIR
          ENV
          FFTW_ROOT
          ENV
          FFTW3_ROOT
    NO_DEFAULT_PATH)
endif()

# find includes
find_path(
  FFTW3_INCLUDE_DIRS
  NAMES "fftw3.h"
  PATH_SUFFIXES "include"
  PATHS ${FFTW_ROOT}
  HINTS ENV
        FFTW_DIR
        ENV
        FFTW3_DIR
        ENV
        FFTW_ROOT
        ENV
        FFTW3_ROOT
  NO_DEFAULT_PATH)

set(FFTW3_LIBRARIES ${FFTW3_LIB})

if (FFTW3_INCLUDE_DIRS)
find_package_handle_standard_args(Fftw DEFAULT_MSG FFTW3_INCLUDE_DIRS
                                  FFTW3_LIBRARIES)
else()
find_package_handle_standard_args(Fftw DEFAULT_MSG FFTW3_LIBRARIES)
endif()
			  set(FFTW3_FOUND ON)
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 INTERFACE IMPORTED)
  if(FFTW3_INCLUDE_DIRS)
    set_target_properties(FFTW3::fftw3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                  "${FFTW3_INCLUDE_DIRS}")
  endif()
  set_target_properties(FFTW3::fftw3 PROPERTIES INTERFACE_LINK_LIBRARIES
                                                "${FFTW3_LIBRARIES}")
endif()

mark_as_advanced(FFTW3_FOUND FFTW3_INCLUDE_DIRS FFTW3_LIBRARIES)
