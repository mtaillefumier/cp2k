/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#ifndef GRID_CPU_TASK_LIST_H
#define GRID_CPU_TASK_LIST_H

#include "../common/grid_basis_set.h"
#include "../common/grid_buffer.h"
#include "../common/grid_constants.h"
#include <stdbool.h>

/*******************************************************************************
 * \brief Allocates a task list for the cpu backend.
 *        See grid_task_list.h for details.
 ******************************************************************************/
void grid_cpu_create_task_list(
		const int ntasks, const int nlevels, const int natoms, const int nkinds,
		const int nblocks, const int *block_offsets,
		const double *atom_positions, const int *atom_kinds,
		const grid_basis_set **basis_sets, const int *level_list,
		const int *iatom_list, const int *jatom_list,
		const int *iset_list, const int *jset_list,
		const int *ipgf_list, const int *jpgf_list,
		const int *border_mask_list, const int *block_num_list,
		const double *radius_list, const double *rab_list,
		void *ptr);

/*******************************************************************************
 * \brief Deallocates given task list, basis_sets have to be freed separately.
 ******************************************************************************/
void grid_cpu_free_task_list(void *ptr);

/*******************************************************************************
 * \brief Collocate all tasks of in given list onto given grids.
 *        See grid_task_list.h for details.
 ******************************************************************************/
void grid_cpu_collocate_task_list(
		void *ptr, const bool orthorhombic,
		const enum grid_func func, const int nlevels,
		const int *npts_global, const int *npts_local,
		const int *shift_local, const int *border_width,
		const double *dh, const double *dh_inv,
		const grid_buffer *pab_blocks, double **grid);

void grid_cpu_integrate_task_list(
		void *const ptr, const bool orthorhombic, const bool compute_tau,
		const int natoms, const int nlevels, const int *npts_global,
		const int *npts_local, const int *shift_local,
		const int *border_width, const double *dh,
		const double *dh_inv, const grid_buffer *const pab_blocks,
		const double **grid, grid_buffer *hab_blocks,
		double *forces, double *virial);

#endif
