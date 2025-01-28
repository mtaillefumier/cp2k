#!-------------------------------------------------------------------------------------------------!
#!   CP2K: A general program to perform molecular dynamics simulations                             !
#!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                  !
#!                                                                                                 !
#!   SPDX-License-Identifier: GPL-2.0-or-later                                                     !
#!-------------------------------------------------------------------------------------------------!

include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)
include(cp2k_utils)

cp2k_set_default_paths(LIBXC "LibXC")

# For LibXC >= 7, the Fortran interface is only libxcf03
pkg_check_modules(CP2K_LIBXC REQUIRED IMPORTED_TARGET GLOBAL libxc>=7)
pkg_check_modules(CP2K_LIBXCF03 REQUIRED IMPORTED_TARGET GLOBAL libxcf03)

find_package_handle_standard_args(
  LibXC DEFAULT_MSG CP2K_LIBXC_FOUND CP2K_LIBXC_LINK_LIBRARIES
  CP2K_LIBXC_INCLUDE_DIRS)

if(CP2K_LIBXC_FOUND)
  if(NOT TARGET cp2k::Libxc::xc)
    add_library(cp2k::Libxc::xc INTERFACE IMPORTED)
  endif()

  if(CP2K_LIBXC_INCLUDE_DIRS)
    set_target_properties(
      cp2k::Libxc::xc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${CP2K_LIBXC_INCLUDE_DIRS}")
  endif()
  target_link_libraries(cp2k::Libxc::xc INTERFACE PkgConfig::CP2K_LIBXCF03
                                                  PkgConfig::CP2K_LIBXC)
endif()

mark_as_advanced(CP2K_LIBXC_FOUND CP2K_LIBXC_LINK_LIBRARIES
                 CP2K_LIBXC_INCLUDE_DIRS)
