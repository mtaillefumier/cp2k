include(FindPackageHandleStandardArgs)
find_package(PkgConfig REQUIRED)

# set(LIBXC_FOUND NO)

if(PKG_CONFIG_FOUND)
  pkg_check_modules(LIBXC libxc>=${LibXC_FIND_VERSION} libxcf03 libxcf90
                    IMPORTED_TARGET)
endif()

if(NOT LIBXC_FOUND)
  find_library(
    LIBXC_LIBRARIES
    NAMES xc
    HINTS ENV
          EBROOTLIBXC
          ENV
          LIBXCROOT
          ENV
          LIBXC_ROOT
          ENV
          LIBXC_PREFIX
    DOC "libxc libraries list")

  find_library(
    LIBXCF90_LIBRARIES
    NAMES xcf90
    HINTS ENV
          EBROOTLIBXC
          ENV
          LIBXCROOT
          ENV
          LIBXC_ROOT
          ENV
          LIBXC_PREFIX
    DOC "fortran 90 binding of the libxc library")

  find_library(
    LIBXCF03_LIBRARIES
    NAMES xcf03
    HINTS ENV
          EBROOTLIBXC
          ENV
          LIBXCROOT
          ENV
          LIBXC_ROOT
          ENV
          LIBXC_PREFIX
    DOC "fortran 2003 binding of the libxc library")

  if(LIBXC_LIBRARIES)
    set(LIBXC_FOUND TRUE)
  endif()
endif()

find_path(
  LIBXC_INCLUDE_DIRS
  NAMES xc.h
  PATH_SUFFIXES libxc
  PATHS ${LIBXC_INCLUDE_DIR}
  HINTS ENV
        EBROOTLIBXC
        ENV
        LIBXCROOT
        ENV
        LIBXC_ROOT
        ENV
        LIBXC_PREFIX)

if(LIBXC_INCLUDE_DIRS)
  find_package_handle_standard_args(LibXC DEFAULT_MSG LIBXC_FOUND
                                    LIBXC_LIBRARIES LIBXC_INCLUDE_DIRS)
else()
  find_package_handle_standard_args(LibXC DEFAULT_MSG LIBXC_FOUND
                                    LIBXC_LIBRARIES)
endif()
if(LIBXC_FOUND AND NOT TARGET Libxc::xc)
  add_library(Libxc::xc INTERFACE IMPORTED)
  if(LIBXC_INCLUDE_DIRS)
    set_target_properties(
      Libxc::xc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXC_INCLUDE_DIRS}"
                           INTERFACE_LINK_LIBRARIES "${LIBXC_LIBRARIES}")
  else()
    set_target_properties(Libxc::xc PROPERTIES INTERFACE_LINK_LIBRARIES
                                               "${LIBXC_LIBRARIES}")
  endif()
endif()

mark_as_advanced(LIBXC_FOUND LIBXC_LIBRARIES LIBXC_INCLUDE_DIRS)
