# find Elpa via pkg-config or easybuild

include(FindPackageHandleStandardArgs)
find_package(PkgConfig)

# First try with pkg-config
if(PKG_CONFIG_FOUND)
  pkg_search_module(
    ELPA
    elpa
    elpa_openmp
    elpa-openmp-2019.05.001
    elpa_openmp-2019.11.001
    elpa_openmp-2020.05.001
    elpa-openmp-2021.11.002
    elpa-2019.05.001
    elpa-2019.11.001
    elpa-2020.05.001
    elpa-2021.11.002)
endif()

if(NOT ELPA_FOUND)
  find_library(
    ELPA_LIBRARIES
    NAMES elpa elpa_openmp
    HINTS ENV
          EBROOTELPA
          ENV
          ELPAROOT
          ENV
          ELPA_ROOT
          ENV
          ORNL_ELPA_ROOT
    DOC "elpa libraries list")
endif()

# always search for the include directory.
find_path(
  ELPA_INCLUDE_DIRS
  NAMES elpa/elpa.h elpa/elpa_constants.h
  PATHS ${ELPA_INCLUDE_DIR}
  PATH_SUFFIXES include/elpa_openmp-$ENV{EBVERSIONELPA}
                include/elpa-$ENV{EBVERSIONELPA}
  HINTS ENV
        ELPAROOT
        ENV
        ELPA_ROOT
        ENV
        ORNL_ELPA_ROOT
        ENV
        EBROOTELPA)

find_package_handle_standard_args(Elpa "DEFAULT_MSG" ELPA_LIBRARIES
                                  ELPA_INCLUDE_DIRS)

if(NOT ELPA_FOUND)
  set(ELPA_FOUND ON)
endif()

if(ELPA_FOUND AND NOT TARGET elpa::elpa)
  add_library(ELPA::elpa INTERFACE IMPORTED)
  set_target_properties(ELPA::elpa PROPERTIES INTERFACE_LINK_LIBRARIES
                                              "${ELPA_LIBRARIES}")
  if(ELPA_INCLUDE_DIRS)
    set_target_properties(ELPA::elpa PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                "${ELPA_INCLUDE_DIRS}")
  endif()
endif()

mark_as_advanced(ELPA_FOUND ELPA_LIBRARIES ELPA_INCLUDE_DIRS)
