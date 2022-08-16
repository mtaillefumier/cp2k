include(FindPackageHandleStandardArgs)
find_package(PkgConfig)

# plumed has a pkg-config module
pkg_search_module(PLUMED "plumed")

find_library(
  PLUMED_LIBRARIES,
  NAMES plumed
  PATH_SUFFIXES lib
  PATHS $ENV{PLUMED_PREFIX}
        $ENV{PLUMEDPREFIX}
        $ENV{PLUMED_ROOT}
        $ENV{PLUMEDROOT}
        $ENV{EBPLUMEDROOT}
        $ENV{EB_PLUMEDROOT}
        $ENV{EB_PLUMED_ROOT})

find_path(
  PLUMED_INCLUDE_DIRS
  NAMES analysis/AnalysisBase.h tools/Angle.h tools/AtomNumber.h
  PATH_SUFFIXES inc include include/plumed plumed
  PATHS $ENV{PLUMED_PREFIX}
        $ENV{PLUMEDPREFIX}
        $ENV{PLUMED_ROOT}
        $ENV{PLUMEDROOT}
        $ENV{EBPLUMEDROOT}
        $ENV{EB_PLUMEDROOT}
        $ENV{EB_PLUMED_ROOT})

find_package_handle_standard_args(Plumed DEFAULT_MSG PLUMED_INCLUDE_DIRS
                                  PLUMED_LIBRARIES)

if(PLUMED_FOUND AND NOT TARGET plumed::plumed)
  add_library(plumed::plumed INTERFACE IMPORTED)
  set_target_properties(
    plumed::plumed
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PLUMED_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES "${PLUMED_LIBRARIES}")
endif()
