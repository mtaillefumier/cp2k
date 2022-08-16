# Copyright ETH Zurich 2022 -
#
include(FindPackageHandleStandardArgs)

find_package(PkgConfig REQUIRED)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(LIBXSMM IMPORTED_TARGET GLOBAL libxsmm QUIET)
  pkg_check_modules(LIBXSMMEXT IMPORTED_TARGET GLOBAL libxsmmext QUIET)
  pkg_check_modules(LIBXSMMF IMPORTED_TARGET GLOBAL libxsmmf QUIET)
  pkg_check_modules(LIBXSMMNOBLAS IMPORTED_TARGET GLOBAL libxsmmnoblas QUIET)
endif()

if(NOT LIBXSMM_FOUND)
  find_library(
    LIBXSMM_LIBRARIES
    NAMES libxsmm
    PATH_SUFFIXES lib lib64
    HINTS ENV
          EBROOTLIBXSMM
          ENV
          LIBXSMM_DIR
          ENV
          LIBXSMMROOT
          ENV
          LIBXSMM_ROOT
    DOC "libxsmm libraries list")
endif()

if(NOT LIBXSMMF_FOUND)
  find_library(
    LIBXSMMF_LIBRARIES
    NAMES libxsmmf
    PATH_SUFFIXES lib lib64
    HINTS ENV
          EBROOTLIBXSMM
          ENV
          LIBXSMM_DIR
          ENV
          LIBXSMMROOT
          ENV
          LIBXSMM_ROOT
    DOC "libxsmm libraries list")
endif()

if(NOT LIBXSMMEXT_FOUND)
  find_library(
    LIBXSMMEXT_LIBRARIES
    NAMES libxsmmext
    PATH_SUFFIXES lib lib64
    HINTS ENV
          EBROOTLIBXSMM
          ENV
          LIBXSMM_DIR
          ENV
          LIBXSMMROOT
          ENV
          LIBXSMM_ROOT
    DOC "libxsmm libraries list")
endif()

if(NOT LIBXSMMNOBLAS_FOUND)
  find_library(
    LIBXSMM_NOBLAS_LIBRARIES
    NAMES libxsmmnoblas
    PATH_SUFFIXES lib lib64
    HINTS ENV
          EBROOTLIBXSMM
          ENV
          LIBXSMM_DIR
          ENV
          LIBXSMMROOT
          ENV
          LIBXSMM_ROOT
    DOC "libxsmm libraries list")
endif()

find_path(
  LIBXSMM_INCLUDE_DIRS
  NAMES libxsmm.h libxsmm.mod
  PATH_SUFFIXES libxsmm
  HINTS ENV XSMM_ROOT ENV XSMM_DIR ENV XSMMROOT
  DOC "libxsmm include directory")

if(LIBXSMM_INCLUDE_DIRS)
  find_package_handle_standard_args(
    LibXSMM
    DEFAULT_MSG
    LIBXSMM_INCLUDE_DIRS
    LIBXSMMNOBLAS_LIBRARIES
    LIBXSMMEXT_LIBRARIES
    LIBXSMMF_LIBRARIES
    LIBXSMM_LIBRARIES)
else()
  find_package_handle_standard_args(
    LibXSMM DEFAULT_MSG LIBXSMMNOBLAS_LIBRARIES LIBXSMMEXT_LIBRARIES
    LIBXSMMF_LIBRARIES LIBXSMM_LIBRARIES)
  set(LIBXSMM_INCLUDE_DIRS "/usr/include")
endif()

if(NOT TARGET LibXSMM::libxsmm)
  add_library(LibXSMM::libxsmm INTERFACE IMPORTED)
  add_library(LibXSMM::libxsmmf INTERFACE IMPORTED)
  add_library(LibXSMM::libxsmmext INTERFACE IMPORTED)
  add_library(LibXSMM::libxsmmnoblas INTERFACE IMPORTED)
  set_target_properties(LibXSMM::libxsmm PROPERTIES INTERFACE_LINK_LIBRARIES
                                                    "${LIBXSMM_LIBRARIES}")
  set_target_properties(
    LibXSMM::libxsmmf PROPERTIES INTERFACE_LINK_LIBRARIES
                                 "${LIBXSMMF_LIBRARIES};${LIBXSMM_LIBRARIES}")
  set_target_properties(
    LibXSMM::libxsmmext
    PROPERTIES INTERFACE_LINK_LIBRARIES
               "${LIBXSMMEXT_LIBRARIES};${LIBXSMM_LIBRARIES}")
  set_target_properties(
    LibXSMM::libxsmmnoblas
    PROPERTIES INTERFACE_LINK_LIBRARIES
               "${LIBXSMMNOBLAS_LIBRARIES};${LIBXSMM_LIBRARIES}")

  set_target_properties(
    LibXSMM::libxsmm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                "${LIBXSMM_INCLUDE_DIRS}")
endif()

mark_as_advanced(LIBXSMM_INCLUDE_DIRS LIBXSMMNOBLAS_LIBRARIES
                 LIBXSMMEXT_LIBRARIES LIBXSMMF_LIBRARIES LIBXSMM_LIBRARIES)
