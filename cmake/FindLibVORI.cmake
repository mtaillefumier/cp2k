include(FindPackageHandleStandardArgs)

find_library(
  LIBVORI_LIBRARIES
  NAMES vori
  HINTS ENV
        EBLIBVORI
        ENV
        LIBVORIROOT
        ENV
        LIBVORI_ROOT
        ENV
        LIBVORI_PREFIX
  DOC "libvori library")

find_path(
  LIBVORI_INCLUDE_DIRS
  NAMES voro++.h
  PATH_SUFFIXES vori
  HINTS ENV
        EBLIBVORI
        ENV
        LIBVORIROOT
        ENV
        LIBVORI_ROOT
        ENV
        LIBVORI_PREFIX
  DOC "libvori header files directory")

if(LIBVORI_INCLUDE_DIRS)
  find_package_handle_standard_args(LibVORI DEFAULT_MSG LIBVORI_LIBRARIES
                                    LIBVORI_INCLUDE_DIRS)
else()
  find_package_handle_standard_args(LibVORI DEFAULT_MSG LIBVORI_LIBRARIES)
endif()
set(LIBVORI_FOUND ON)
if(LIBVORI_FOUND AND NOT TARGET VORI::vori)
  add_library(VORI::vori INTERFACE IMPORTED)
  set_target_properties(VORI::vori PROPERTIES INTERFACE_LINK_LIBRARIES
                                              "${LIBVORI_LIBRARIES}")
  if(LIBVORI_INCLUDE_DIRS)
    set_target_properties(VORI::vori PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                "${LIBVORI_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(LIBVORI_INCLUDE_DIRS LIBVORI_LIBRARIES)
