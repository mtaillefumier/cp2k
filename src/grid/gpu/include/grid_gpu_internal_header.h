/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#ifndef GRID_GPU_INTERNAL_HEADER_H
#define GRID_GPU_INTERNAL_HEADER_H

/*******************************************************************************
 * \brief Internal representation of a task.
 * \author Ole Schuett
 ******************************************************************************/
struct gpu_task {
  int level;
  int iatom;
  int jatom;
  int iset;
  int jset;
  int ipgf;
  int jpgf;
  int border_mask;
  int block_num;
  double radius;
  double rab[3];
};

/*******************************************************************************
 * \brief Internal representation of a task list.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct  {
  int ntasks;
  int nlevels;
  int natoms;
  int nkinds;
  int nblocks;
  int *tasks_per_level;
  cudaStream_t *level_streams;
  cudaStream_t main_stream;
  int lmax;
  int stats[2][20]; // [has_border_mask][lp]
  // device pointers
  int *block_offsets_dev;
  double *atom_positions_dev;
  int *atom_kinds_dev;
  grid_basis_set *basis_sets_dev;
  grid_gpu_task *tasks_dev;
  double **grid_dev;
  size_t *grid_dev_size;
  int *tasks_per_level;
} grid_gpu_task_list;
#endif
