#!-------------------------------------------------------------------------------------------------!
#!   CP2K: A general program to perform molecular dynamics simulations                             !
#!   Copyright 2000-2025 CP2K developers group <https://cp2k.org>                                  !
#!                                                                                                 !
#!   SPDX-License-Identifier: GPL-2.0-or-later                                                     !
#!-------------------------------------------------------------------------------------------------!

include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)
include(cp2k_utils)

cp2k_set_default_paths(GRPP "LIBGRPP")

if(PKG_CONFIG_FOUND)
  pkg_check_modules(CP2K_GRPP IMPORTED_TARGET GLOBAL libgrpp)
endif()

find_package_handle_standard_args(GRPP DEFAULT_MSG CP2K_GRPP_INCLUDE_DIRS
                                  CP2K_GRPP_FOUND CP2K_GRPP_LINK_LIBRARIES)

if(CP2K_GRPP_FOUND)
  if(NOT TARGET cp2k::grpp::grpp)
    add_library(cp2k::grpp::grpp INTERFACE IMPORTED)
  endif()

  target_link_libraries(cp2k::grpp::grpp INTERFACE PkgConfig::CP2K_GRPP)
endif()

mark_as_advanced(CP2K_GRPP_FOUND CP2K_GRPP_LINK_LIBRARIES
                 CP2K_GRPP_INCLUDE_DIRS)
