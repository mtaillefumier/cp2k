# find spglib if in non-standard location set environment variabled `SPG_DIR` to
# the root directory

include(FindPackageHandleStandardArgs)

find_package(PkgConfig REQUIRED)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(LIBSPG spglib QUIET)
endif()

if(NOT LIBSPG_FOUND)
  find_library(
    LIBSPG_LIBRARIES
    NAMES symspg
    HINTS ENV EBROOTSPGLIB ENV SPG_DIR ENV LIBSPGROOT
    DOC "spglib libraries list")

  if(NOT LIBSPG_LIBRARIES)
    set(LIBSPG_FOUND ON)
  endif()
endif()

find_path(
  LIBSPG_INCLUDE_DIRS
  NAMES spglib.h
  HINTS ENV EBROOTSPGLIB ENV SPG_DIR ENV LIBSPGROOT
  DOC "spglib include directory")

if(LIBSPG_INCLUDE_DIRS)
  find_package_handle_standard_args(LibSPG DEFAULT_MSG LIBSPG_LIBRARIES
                                    LIBSPG_INCLUDE_DIRS)
else()
  find_package_handle_standard_args(LibSPG DEFAULT_MSG LIBSPG_LIBRARIES)
endif()
if(LIBSPG_FOUND AND NOT TARGET LIBSPG::libspg)
  add_library(LIBSPG::libspg INTERFACE IMPORTED)
  set_target_properties(LIBSPG::libspg PROPERTIES INTERFACE_LINK_LIBRARIES
                                                  "${LIBSPG_LIBRARIES}")
  if(LIBSPG_INCLUDE_DIRS)
    set_target_properties(
      LIBSPG::libspg PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${LIBSPG_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(LIBSPG_LIBRARIES)
mark_as_advanced(LIBSPG_INCLUDE_DIRS)
mark_as_advanced(LIBSPG_FOUND)
