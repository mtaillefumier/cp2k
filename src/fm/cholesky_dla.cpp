//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <blas/util.hh>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/auxiliary/norm.h"
#include "dlaf/blas/tile.h"
#include "dlaf/common/format_short.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/error.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/factorization/cholesky.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf/matrix/print_numpy.h"

using pika::execution::experimental::keep_future;
using pika::execution::experimental::start_detached;
using pika::execution::experimental::when_all;

using dlaf::Backend;
using dlaf::Coord;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::GlobalTileIndex;
using dlaf::LocalTileIndex;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementIndex;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::comm::Index2D;
using dlaf::common::Ordering;
using dlaf::internal::transformDetach;
using dlaf::matrix::MatrixMirror;

static bool dlaf_init_ = false;

// Cblacs does not seem to have public headers files and is only used in
// scalapack.

extern "C" MPI_Comm Cblacs2sys_handle(int ictxt);
extern "C" void Cblacs_get(int ictxt, int inum, int *comm);
extern "C" void Cblacs_gridinfo(int ictxt, int *np, int *mp, int *px, int *py);

// queries the grid blacs context to get the communication blacs context
static int get_comm_context(const int grid_context) {
  int comm_context;
  Cblacs_get(grid_context, 10, &comm_context);
  return comm_context;
}

// gets MPI_Comm from the grid blacs context
static MPI_Comm get_communicator(const int grid_context) {
  int comm_context = get_comm_context(grid_context);
  MPI_Comm comm = Cblacs2sys_handle(comm_context);
  return comm;
}

static int get_grid_context(int *desca)
{
  return desca[1];
}

extern "C" void dlaf_init() {
  if (!dlaf_init_) {
    using namespace pika::program_options;
    int argc = 2;

    std::string tmp = "--pika:threads=";
    auto my_string = getenv("OMP_NUM_THREADS");
    if(my_string == nullptr) {
      tmp += std::to_string(1);
    } else {
      tmp += std::string(my_string);
    }

    // TODO something wrong with passing command line arguments by hand
    const char *argv[] = {"--pika:print-bind", tmp.c_str(), nullptr};
    options_description desc_commandline("cp2k_dlaf");
    //desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
    desc_commandline.add(dlaf::getOptionsDescription());
    
    /* pika initialization. Last parameter to be added. */
    pika::init_params p;
    p.desc_cmdline = desc_commandline;
    p.rp_callback = dlaf::initResourcePartitionerHandler;
    pika::start(nullptr, argc, argv, p);
    const int argc_dla = 1;
    const char *argv_dla[] = {"cp2k_dlaf", nullptr};
    dlaf::initialize(argc_dla, argv_dla);
    dlaf_init_ = true;
    pika::suspend();
  }
}

extern "C" void dlaf_finalize() {
  pika::resume();
  pika::async([]{pika::finalize();});
  dlaf::finalize();
  pika::stop();
  dlaf_init_ = false;
}

template <typename T> void pxpotrf_dla(char uplo__, int n__, T *a__, int ia__, int ja__, int *desca__, int &info__)
{

  if (uplo__ != 'U' && uplo__ != 'u' && uplo__ != 'L' && uplo__ != 'l') {
    std::cerr << "DLA Cholesky : The UpLo parameter has a incorrect value. Please check the scalapack documentation.\n";
    info__ = -1;
    return;
  }

  // need to extract this from the scalapack descriptor
  if (desca__[0] != 1) {
    // only treat dense matrices
    info__ = -1;
    return;
  }

  if (!dlaf_init_) {
    std::cout << "Error: DLA Future must be initialized\n";
    info__ = -1;
  }

  using dlaf::common::make_data;

  pika::resume();
  // matrix sizes
  int m, n;

  // block sizes
  int nb, mb;
  
  // retrive the matrix sizes
  m = desca__[3];
  n = desca__[2];

  nb = desca__[5];
  mb = desca__[4];

  // TODO
  // DONE - dlaf initialization
  //      - matrix mirror
  // DONE - resume suspend pika runtime
  //      - general cleanup
  // DONE - fortran interface uplo
  //      - cblcs call

  int np, mp, size;
  MPI_Comm comm = get_communicator(desca__[1]);
  Communicator world(comm);
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  int coords[2] = {-1, -1};
  MPI_Comm_size(comm, &size);
  Cblacs_gridinfo(desca__[1], dims, dims + 1, coords, coords + 1);

  CommunicatorGrid comm_grid(world, dims[0], dims[1], Ordering::RowMajor);
  
  // Allocate memory for the matrix
  GlobalElementSize matrix_size(n, m);
  TileElementSize block_size(nb, mb);
  Index2D src_rank_index(0,0);
  dlaf::matrix::Distribution distribution(matrix_size,
                                          block_size,
                                          comm_grid.size(),
                                          comm_grid.rank(),
                                          src_rank_index);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // leading dimension
  const int ld_ = desca__[8];

  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, ld_);
  Matrix<T, Device::CPU> mat(std::move(distribution), layout, a__);

  {
    using MatrixMirrorType = MatrixMirror<T,  dlaf::Device::Default, Device::CPU>;
    MatrixMirrorType matrix(mat);

    switch(uplo__) {
     case 'U':
     case 'u':
       dlaf::factorization::cholesky<Backend::Default, Device::Default, T>(comm_grid, blas::Uplo::Upper, matrix.get());
       break;
    case 'L':
    case 'l':
      dlaf::factorization::cholesky<Backend::Default, Device::Default, T>(comm_grid, blas::Uplo::Lower, matrix.get());
      break;
    default:
      break;
    }

    // TODO: Can this be relaxed (removed)?
    matrix.get().waitLocalTiles();
  }
  
  // TODO: Can this be relaxed (removed)?
  mat.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  
  pika::suspend();
  info__ = 0;
}

extern "C" void pdpotrf_dlaf_(char *uplo__, int n__, double *a__, int ia__, int ja__, int *desca__, int *info__)
{
  pxpotrf_dla<double>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}

extern "C" void pspotrf_dlaf_(char *uplo__, int n__, float *a__, int ia__, int ja__, int *desca__, int *info__)
{
  pxpotrf_dla<float>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}

extern "C" void pdpotrf_dlaf(char *uplo__, int n__, double *a__, int ia__, int ja__, int *desca__, int *info__)
{
  pxpotrf_dla<double>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}

extern "C" void pspotrf_dlaf(char *uplo__, int n__, float *a__, int ia__, int ja__, int *desca__, int *info__)
{
  pxpotrf_dla<float>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}
