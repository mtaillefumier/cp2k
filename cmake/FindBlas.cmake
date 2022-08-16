# ETH Zurich 2022 -
#
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindBLAS
--------

Find Basic Linear Algebra Subprograms (BLAS) library

This module finds an installed Fortran library that implements the
BLAS linear-algebra interface (see http://www.netlib.org/blas/).

The approach follows that taken for the ``autoconf`` macro file, ``acx_blas.m4``
(distributed at http://ac-archive.sourceforge.net/ac-archive/acx_blas.html) with
modularization in mind. Adding a new blas implementation can be done by creating
a FindNewBlas.cmake which should have a target NewBLAS::blas available and
adding the shiny NewBLAS to

Input Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``BLAS_STATIC``
  if ``ON`` use static linkage

``BLAS_VENDOR``
  If set, checks only the specified vendor, if not set checks all the
  possibilities.  List of vendors valid in this module:

  * ``MKL``
  * ``OpenBLAS``
  * ``GENERIC``
  * ``SGI``
  * ``FLAME``
  * ``ATLAS``
  * ``ACML``
  * ``ACML_MP``
  * ``ACML_GPU``
  * ``Armpl``

``BLAS_THREADING``
  select the type of threading support the blas library should use

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``BLAS_FOUND``
  library implementing the BLAS interface is found

``BLAS_LINKER_FLAGS``
  uncached list of required linker flags (excluding ``-l`` and ``-L``).

``BLAS_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  to use BLAS (may be empty if compiler implicitly links BLAS)

``BLAS_INCLUDE_DIRS``
  uncached list of include directories (using full path name) to give to the compiler

``BLAS95_LIBRARIES``
  uncached list of libraries (using full path name) to link against
  to use BLAS95 interface

``BLAS_INCLUDE_DIRS``
  uncached list of include directories

  the module also exports a set of targets that can be used by cmake. These
  target are exposed through

  blas::blas

.. note::

  C, CXX or Fortran must be enabled to detect a BLAS library.
  C or CXX must be enabled to use Intel Math Kernel Library (MKL).

  For example, to use Intel MKL libraries and/or Intel compiler:

  .. code-block:: cmake

    set(BLAS_VENDOR MKL)
    find_package(BLAS)

Hints
^^^^^

Set the ``MKLROOT`` environment variable to a directory that contains an MKL
installation, or add the directory to the dynamic library loader environment
variable for your platform (``LIB``, ``DYLD_LIBRARY_PATH`` or
``LD_LIBRARY_PATH``).

#]=======================================================================]

if(NOT
   (CMAKE_C_COMPILER_LOADED
    OR CMAKE_CXX_COMPILER_LOADED
    OR CMAKE_Fortran_COMPILER_LOADED))
  message(FATAL_ERROR "FindBLAS requires Fortran, C, or C++ to be enabled.")
endif()

set(BLAS_VENDOR_LIST
    "Any"
    "MKL"
    "OpenBLAS"
    "SCI"
    "GenericBLAS"
    "Arm"
    "FlexiBLAS"
    "ATLAS")

set(__BLAS_VENDOR_LIST "MKL" "OpenBLAS" "SCI" "GenericBLAS" "Arm" "FlexiBLAS" "ATLAS")

set(BLAS_VENDOR
    "Any"
    CACHE STRING "Blas library for computations on host")
set_property(CACHE BLAS_VENDOR PROPERTY STRINGS ${BLAS_VENDOR_LIST})

if(NOT ${BLAS_VENDOR} IN_LIST BLAS_VENDOR_LIST)
  message(FATAL_ERROR "Invalid Host BLAS backend")
endif()

set(BLAS_THREAD_LIST "sequential" "thread" "gnu-thread" "intel-thread"
                     "tbb-thread" "openmp")

set(BLAS_THREADING
    "thread"
    CACHE STRING "threaded blas library")
set_property(CACHE BLAS_THREADING PROPERTY STRINGS ${BLAS_THREAD_LIST})

if(NOT ${BLAS_THREADING} IN_LIST BLAS_THREAD_LIST)
  message(FATAL_ERROR "Invalid threaded BLAS backend")
endif()

set(BLAS_INTERFACE_BITS_LIST "32bits" "64bits")
set(BLAS_INTERFACE
    "32bits"
    CACHE STRING
          "32 bits integers are used for indices, matrices and vectors sizes")
set_property(CACHE BLAS_INTERFACE PROPERTY STRINGS ${BLAS_INTERFACE_BITS_LIST})

if(NOT ${BLAS_INTERFACE} IN_LIST BLAS_INTERFACE_BITS_LIST)
  message(
    FATAL_ERROR
      "Invalid parameters. Blas and lapack can exist in two flavors 32 or 64 bits interfaces (relevant mostly for mkl)"
  )
endif()

set(BLAS_FOUND FALSE)

# first check for a specific implementation if requested

if(NOT BLAS_VENDOR MATCHES "Any")
  find_package(${BLAS_VENDOR} REQUIRED)
  if(TARGET ${BLAS_VENDOR}::blas)
    add_library(BLAS::blas INTERFACE IMPORTED)
    get_target_property(BLAS_INCLUDE_DIRS ${BLAS_VENDOR}::blas
                        INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(BLAS_LIBRARIES ${BLAS_VENDOR}::blas
                        INTERFACE_LINK_LIBRARIES)
    set_target_properties(
      BLAS::blas PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
                            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
    set(BLAS_FOUND TRUE)
  endif()
endif()

# search for any blas implementation
if(NOT TARGET BLAS::blas AND NOT BLAS_FOUND)
  foreach(_libs ${__BLAS_VENDOR_LIST})
    if(NOT TARGET BLAS::blas)
      # i exclude the first item of the list
      find_package(${_libs})
      if(TARGET ${_libs}::blas AND NOT TARGET BLAS::blas)
        add_library(BLAS::blas INTERFACE IMPORTED)
        get_target_property(BLAS_INCLUDE_DIRS ${_libs}::blas
                            INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(BLAS_LIBRARIES ${_libs}::blas
                            INTERFACE_LINK_LIBRARIES)
        set_target_properties(
          BLAS::blas
          PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${BLAS_INCLUDE_DIRS}"
                     INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
        set(BLAS_VENDOR "${_libs}")
        set(BLAS_FOUND TRUE)
      endif()
    endif()
  endforeach()
endif()

find_package_handle_standard_args(
  Blas REQUIRED_VARS BLAS_LIBRARIES BLAS_INCLUDE_DIRS BLAS_VENDOR)

mark_as_advanced(BLAS_INCLUDE_DIRS)
mark_as_advanced(BLAS_LIBRARIES)
mark_as_advanced(BLAS_VENDOR)
mark_as_advanced(BLAS_FOUND)
