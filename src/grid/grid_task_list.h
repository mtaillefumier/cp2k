/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/
#ifndef GRID_TASK_LIST_H
#define GRID_TASK_LIST_H

#include <stdbool.h>

#include "common/grid_basis_set.h"
#include "common/grid_constants.h"

void grid_create_task_list(
		const int ntasks, const int nlevels, const int natoms, const int nkinds,
		const int nblocks, const int block_offsets[nblocks],
		const double atom_positions[natoms][3], const int atom_kinds[natoms],
		const grid_basis_set *basis_sets[nkinds], const int level_list[ntasks],
		const int iatom_list[ntasks], const int jatom_list[ntasks],
		const int iset_list[ntasks], const int jset_list[ntasks],
		const int ipgf_list[ntasks], const int jpgf_list[ntasks],
		const int border_mask_list[ntasks], const int block_num_list[ntasks],
		const double radius_list[ntasks], const double rab_list[ntasks][3],
		void *ptr);

/*******************************************************************************
 * \brief Deallocates given task list, basis_sets have to be freed separately.
 * \author Ole Schuett
 ******************************************************************************/
void grid_free_task_list(void *ptr);

/*******************************************************************************
 * \brief Collocate all tasks of in given list onto given grids.
 *
 * \param task_list       Task list to collocate.
 * \param orthorhombic    Whether simulation box is orthorhombic.
 * \param func            Function to be collocated, see grid_prepare_pab.h
 * \param nlevels         Number of grid levels.
 *
 *      The remaining params are given for each grid level:
 *
 * \param npts_global     Number of global grid points in each direction.
 * \param npts_local      Number of local grid points in each direction.
 * \param shift_local     Number of points local grid is shifted wrt global grid
 * \param border_width    Width of halo region in grid points in each direction.
 * \param dh              Incremental grid matrix.
 * \param dh_inv          Inverse incremental grid matrix.
 * \param pab_blocks      Buffer that contains the density matrix blocks.
 * \param grid            The output grid array to collocate into.
 *
 * \author Ole Schuett
 ******************************************************************************/
void grid_collocate_task_list(
		void *ptr, const bool orthorhombic,
		const enum grid_func func, const int nlevels,
		const int npts_global[nlevels][3], const int npts_local[nlevels][3],
		const int shift_local[nlevels][3], const int border_width[nlevels][3],
		const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
		const grid_buffer *pab_blocks, double *grid[nlevels]);

/*******************************************************************************
 * \brief Integrate all tasks of in given list from given grids.
 *
 * \param task_list        Task list to collocate.
 * \param orthorhombic     Whether simulation box is orthorhombic.
 * \param compute_tau      When true then <nabla a| V | nabla b> is computed.
 * \param natoms           Number of atoms.
 * \param nlevels          Number of grid levels.
 *
 *      The remaining params are given for each grid level:
 *
 * \param npts_global     Number of global grid points in each direction.
 * \param npts_local      Number of local grid points in each direction.
 * \param shift_local     Number of points local grid is shifted wrt global grid
 * \param border_width    Width of halo region in grid points in each direction.
 * \param dh              Incremental grid matrix.
 * \param dh_inv          Inverse incremental grid matrix.
 * \param grid            Grid array to integrate from.
 *
 * \param pab_blocks      Optional density blocks, needed for forces and virial.
 *
 * \param hab_blocks      Output buffer with the Hamiltonian matrix blocks.
 * \param forces          Optional output forces, requires pab_blocks.
 * \param virial          Optional output virials, requires pab_blocks.
 *
 * \author Ole Schuett
 ******************************************************************************/
void grid_integrate_task_list(
		void *ptr, const bool orthorhombic,
		const bool compute_tau, const int natoms, const int nlevels,
		const int npts_global[nlevels][3], const int npts_local[nlevels][3],
		const int shift_local[nlevels][3], const int border_width[nlevels][3],
		const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3],
		const grid_buffer *pab_blocks, const double *grid[nlevels],
		grid_buffer *hab_blocks, double forces[natoms][3], double virial[3][3]);

#endif

// EOF
