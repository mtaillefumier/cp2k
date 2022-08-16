# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLAPACK
----------

Find Linear Algebra PACKage (LAPACK) library

This module finds an installed Fortran library that implements the
LAPACK linear-algebra interface (see http://www.netlib.org/lapack/).

Input Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``BLAS_STATIC``
  if ``ON`` use static linkage

  ``BLAS_THREADS``
  if set, check for threading support. List of possible options
  * ``sequential``
  * ``thread``
  * ``gnu-thread``
  * ``openmp``
  * ``tbb``

  tbb is specific to Intel MKL. thread and gnu-thread are synonimous in most
  systems, openmp is used mostly in openblas, sequential is defined for all
  libraries

``BLAS_VENDOR``
  If set, checks only the specified vendor, if not set checks all the
  possibilities.  List of vendors valid in this module:

  * ``OpenBLAS``
  * ``FLAME``
  * ``MKL``
  * ``ACML``
  * ``Generic``

``BLAS_F95``
  if ``ON`` tries to find the BLAS95/LAPACK95 interfaces

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LAPACK_FOUND``
  library implementing the LAPACK interface is found
``LAPACK_LINKER_FLAGS``
  uncached list of required linker flags (excluding ``-l`` and ``-L``).
``LAPACK_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  to use LAPACK
``LAPACK95_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  to use LAPACK95
``LAPACK95_FOUND``
  library implementing the LAPACK95 interface is found

.. note::

  C, CXX or Fortran must be enabled to detect a BLAS/LAPACK library.
  C or CXX must be enabled to use Intel Math Kernel Library (MKL).

  For example, to use Intel MKL libraries and/or Intel compiler:

  .. code-block:: cmake

  set(BLAS_VENDOR "Generic")
  find_package(LAPACK)
#]=======================================================================]

find_package(PkgConfig)

# check for blas first. Most of the vendor libraries bundle lapack and blas in
# the same library. (MKL, and OPENBLAS do)

set(LAPACK_FOUND FALSE)
set(LAPACK95_FOUND FALSE)
set(LAPACK_LINKER_FLAGS)
set(LAPACK_LIBRARIES)
set(LAPACK_INCLUDE_DIRS)
set(LAPACK95_LIBRARIES)
find_package(PkgConfig)

find_package(Blas REQUIRED)

if(BLAS_FOUND)
  # LAPACK in the Intel MKL 10+ library?
  if(BLAS_VENDOR MATCHES "MKL"
     OR BLAS_VENDOR MATCHES "OpenBLAS"
     OR BLAS_VENDOR MATCHES "Arm"
     OR BLAS_VENDOR MATCHES "SCI")
    # we just need to create the interface that's all
    set(LAPACK_FOUND TRUE)
    get_target_property(LAPACK_INCLUDE_DIRS BLAS::blas
                        INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(LAPACK_LIBRARIES BLAS::blas INTERFACE_LINK_LIBRARIES)
  else()

    # we might get lucky to find a pkgconfig package for lapack (fedora provides
    # one for instance)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(_lapack lapack)
    endif()

    if(NOT _lapack_FOUND)
      find_library(
        LAPACK_LIBRARIES
        NAMES "lapack" "lapack64"
        PATH_SUFFIXES "openblas" "openblas64" "openblas-pthread"
                      "openblas-openmp" "lib" "lib64"
        NO_DEFAULT_PATH)
    else()
      set(LAPACK_LIBRARIES ${_lapack_LIBRARIES})
      set(LPACK_INCLUDE_DIRS ${_lapack_INCLUDE_DIRS})
    endif()
  endif()
endif()

# check if found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Lapack REQUIRED_VARS LAPACK_LIBRARIES)

if(LAPACK_FOUND)
  if(NOT TARGET LAPACK::lapack)
    add_library(LAPACK::lapack INTERFACE IMPORTED)
    set_property(TARGET LAPACK::lapack PROPERTY INTERFACE_LINK_LIBRARIES
                                                ${LAPACK_LIBRARIES})
    set_property(TARGET LAPACK::lapack PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                ${LAPACK_INCLUDE_DIRS})
    add_library(LAPACK::LAPACK ALIAS LAPACK::lapack)
  endif()
endif()

# prevent clutter in cache
mark_as_advanced(LAPACK_FOUND LAPACK_LIBRARIES LAPACK_INCLUDE_DIRS)
