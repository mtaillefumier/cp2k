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
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/error.h>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/init.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/layout_info.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/types.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

// TODO: Remove once https://github.com/eth-cscs/DLA-Future/pull/668 is
// merged.
#include <mkl_service.h>
#include <omp.h>

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

static int get_grid_context(int *desca) { return desca[1]; }

extern "C" void dlaf_init() {
  if (!dlaf_init_) {
    int argc = 1;
    const char *const argv[] = {"cp2k", nullptr};

    pika::program_options::options_description desc("cp2k");
    desc.add(dlaf::getOptionsDescription());

    /* pika initialization */
    pika::init_params p;
    p.rp_callback = dlaf::initResourcePartitionerHandler;
    p.desc_cmdline = desc;
    pika::start(nullptr, argc, argv, p);

    /* DLA-Future initialization */
    dlaf::initialize(argc, argv);
    dlaf_init_ = true;

    pika::suspend();
  }
}

extern "C" void dlaf_finalize() {
  pika::resume();
  pika::async([] { pika::finalize(); });
  dlaf::finalize();
  pika::stop();
  dlaf_init_ = false;
}

class single_threaded_omp {
public:
  single_threaded_omp() : old_threads(mkl_get_max_threads()) {
    mkl_set_num_threads(1);
  }
  ~single_threaded_omp() { mkl_set_num_threads(old_threads); }

private:
  int old_threads;
};

template <typename T>
void pxpotrf_dla(char uplo__, int n__, T *a__, int ia__, int ja__, int *desca__,
                 int &info__) {
  if (uplo__ != 'U' && uplo__ != 'u' && uplo__ != 'L' && uplo__ != 'l') {
    std::cerr << "DLA Cholesky : The UpLo parameter has a incorrect value. "
                 "Please check the scalapack documentation.\n";
    info__ = -1;
    return;
  }

  // need to extract this from the scalapack descriptor
  if (desca__[0] != 1) {
    // only treat dense matrices
    info__ = -1;
    std::cerr << "Error: DLA Future should only treat dense matrices\n";
    return;
  }

  if (!dlaf_init_) {
    std::cerr << "Error: DLA Future must be initialized\n";
    info__ = -1;
  }

  single_threaded_omp sto{};

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
  // DONE - matrix mirror
  // DONE - resume suspend pika runtime
  //      - general cleanup
  // DONE - fortran interface uplo
  //      - cblcs call
  //      - remove omp/mkl_set_num_threads calls (will be handled by DLAF
  //        in https://github.com/eth-cscs/DLA-Future/pull/668)

  int np, mp, size;
  MPI_Comm comm = get_communicator(desca__[1]);
  dlaf::comm::Communicator world(comm);
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  int coords[2] = {-1, -1};
  MPI_Comm_size(comm, &size);
  Cblacs_gridinfo(desca__[1], dims, dims + 1, coords, coords + 1);

  dlaf::comm::CommunicatorGrid comm_grid(world, dims[0], dims[1],
                                         dlaf::common::Ordering::RowMajor);

  // Allocate memory for the matrix
  dlaf::GlobalElementSize matrix_size(n, m);
  dlaf::TileElementSize block_size(nb, mb);
  dlaf::comm::Index2D src_rank_index(0, 0);
  dlaf::matrix::Distribution distribution(matrix_size, block_size,
                                          comm_grid.size(), comm_grid.rank(),
                                          src_rank_index);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  // leading dimension
  const int ld_ = desca__[8];

  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, ld_);
  dlaf::matrix::Matrix<T, dlaf::Device::CPU> mat(std::move(distribution),
                                                 layout, a__);

  {
    dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>
        matrix(mat);

    switch (uplo__) {
    case 'U':
    case 'u':
      dlaf::factorization::cholesky<dlaf::Backend::Default,
                                    dlaf::Device::Default, T>(
          comm_grid, blas::Uplo::Upper, matrix.get());
      break;
    case 'L':
    case 'l':
      dlaf::factorization::cholesky<dlaf::Backend::Default,
                                    dlaf::Device::Default, T>(
          comm_grid, blas::Uplo::Lower, matrix.get());
      break;
    default:
      break;
    }
  }

  pika::suspend();

  info__ = 0;
}

extern "C" void pdpotrf_dlaf_(char *uplo__, int n__, double *a__, int ia__,
                              int ja__, int *desca__, int *info__) {
  pxpotrf_dla<double>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}

extern "C" void pspotrf_dlaf_(char *uplo__, int n__, float *a__, int ia__,
                              int ja__, int *desca__, int *info__) {
  pxpotrf_dla<float>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}

extern "C" void pdpotrf_dlaf(char *uplo__, int n__, double *a__, int ia__,
                             int ja__, int *desca__, int *info__) {
  pxpotrf_dla<double>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}

extern "C" void pspotrf_dlaf(char *uplo__, int n__, float *a__, int ia__,
                             int ja__, int *desca__, int *info__) {
  pxpotrf_dla<float>(*uplo__, n__, a__, ia__, ja__, desca__, *info__);
}
