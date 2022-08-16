#[=======================================================================[.rst:
FindSuperLU
--------

Find the SuperLU library

Input Variables
^^^^^^^^^^^^^^^

``SUPERLU_PREFER_PKGCONFIG``
  if set ``pkg-config`` will be used to search for a BLAS library first
  and if one is found that is preferred

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SUPERLU_FOUND``
  library implementing the BLAS interface is found
``SUPERLU_LINKER_FLAGS``
  uncached list of required linker flags (excluding ``-l`` and ``-L``).
``SUPERLU_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  ``SUPERLU_INCLUDE_DIR``
  uncached list of include directories

.. note::

  C or CXX must be enabled to detect the superlu library.

Hints
^^^^^

Set the ``SUPERLUROOT`` environment variable to a directory that contains
superlu

#]=======================================================================]

include(FindPackageHandleStandardArgs)
find_package(PkgConfig)

pkg_search_module(SUPERLU QUIET "superlu")

if(SUPERLU_INCLUDES AND SUPERLU_LIBRARIES)
  set(SUPERLU_FIND_QUIETLY TRUE)
endif()

find_path(
  SUPERLU_INCLUDE_DIRS
  NAMES supermatrix.h
  PATH_SUFFIXES include include/SuperLU SuperLU superlu include/openmpi-x86_64
                include/mpich-x86_64
  PATHS $ENV{SUPERLU_PREFIX} $ENV{SUPERLU_ROOT} $ENV{SUPERLUROOT}
        $ENV{EBSUPERLUROOT})

find_library(
  SUPERLU_LIBRARIES
  NAMES superlu superlu_dist
  PATH_SUFFIXES lib
  PATHS $ENV{SUPERLU_PREFIX} $ENV{SUPERLU_ROOT} $ENV{SUPERLUROOT}
        $ENV{EBSUPERLUROOT})

find_package_handle_standard_args(SuperLU DEFAULT_MSG SUPERLU_INCLUDE_DIRS
                                  SUPERLU_LIBRARIES)

if(SUPERLU_FOUND AND NOT TARGET superlu::superlu)
  add_library(superlu::superlu INTERFACE IMPORTED)
  set_target_properties(
    superlu::superlu
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SUPERLU_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES "${SUPERLU_LIBRARIES}")
endif()
