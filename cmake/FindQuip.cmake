include(FindPackageHandleStandardArgs)
find_package(PkgConfig)

pkg_search_module(Quip QUIET)

if(QUIP_INCLUDES AND QUIP_LIBRARIES)
  set(QUIP_FIND_QUIETLY TRUE)
endif()

find_path(
  QUIP_INCLUDE_DIRS
  NAMES
  PATH_SUFFIXES include include/quip quip QUIP
  PATHS $ENV{QUIP_PREFIX} $ENV{QUIP_ROOT} $ENV{QUIPROOT} $ENV{EBQUIPROOT})

find_library(
  QUIP_LIBRARIES
  NAMES "quip"
  PATH_SUFFIXES lib lib64
  PATHS $ENV{QUIP_PREFIX} $ENV{QUIP_ROOT} $ENV{QUIPROOT} $ENV{EBQUIPROOT})

find_package_handle_standard_args(Quip DEFAULT_MSG QUIP_INCLUDE_DIRS
                                  QUIP_LIBRARIES)

if(QUIP_FOUND AND NOT TARGET quip::quip)
  add_library(quip::quip INTERFACE IMPORTED)
  set_target_properties(
    quip::quip PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${QUIP_INCLUDE_DIRS}"
                          INTERFACE_LINK_LIBRARIES "${QUIP_LIBRARIES}")
endif()
