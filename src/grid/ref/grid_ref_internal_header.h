/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#ifndef GRID_REF_INTERNAL_HEADER_H
#define GRID_REF_INTERNAL_HEADER_H

#include "../common/grid_basis_set.h"
/*******************************************************************************
 * \brief Internal representation of a task.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
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
} grid_ref_task;

/*******************************************************************************
 * \brief Internal representation of a task list.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
		int ntasks;
		int nlevels;
		int natoms;
		int nkinds;
		int nblocks;
		int *block_offsets;
		double *atom_positions;
		int *atom_kinds;
		grid_basis_set **basis_sets;
		grid_ref_task *tasks;
		int *tasks_per_level;
		int maxco;
} grid_ref_task_list;

#endif
