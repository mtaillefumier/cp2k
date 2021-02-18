/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include "common/grid_context.hpp"

extern "C" {
#include "common/grid_library.h"
#include "ref/grid_ref_task_list.h"
#include "gpu/grid_gpu_task_list.h"
}

static void collocate_test(void *ptr, const bool orthorhombic,
													 const enum grid_func func, const int nlevels,
													 const int *npts_global, const int *npts_local,
													 const int *shift_local, const int *border_width,
													 const double *dh, const double *dh_inv,
													 grid_buffer *pab_blocks, double **grid);
static void integrate_test(	void *ptr, const bool orthorhombic, const bool compute_tau,
														const int natoms, const int nlevels, const int *npts_global,
														const int *npts_local, const int *shift_local,
														const int *border_width, const double *dh,
														const double *dh_inv, grid_buffer *const pab_blocks,
														const double **grid, grid_buffer *hab_blocks, double *forces,
														double *virial);

/* this is the C interface to the grid_context class. The interface
 * differentiate between the reference backend and all the other backends but it
 * will disappear eventually so that the interface is agnostic to the details of
 * the implementation. */


/*******************************************************************************
 * \brief Allocates a task list which can be passed to grid_collocate_task_list.
 *        See grid_task_list.hpp for details.
 ******************************************************************************/
extern "C" void grid_create_task_list(
		const int ntasks, const int nlevels, const int natoms, const int nkinds,
		const int nblocks, const int *block_offsets,
		const double *atom_positions, const int *atom_kinds,
		const grid_basis_set **basis_sets, const int *level_list,
		const int *iatom_list, const int *jatom_list,
		const int *iset_list, const int *jset_list,
		const int *ipgf_list, const int *jpgf_list,
		const int *border_mask_list, const int *block_num_list,
		const double *radius_list, const double *rab_list,
		void *ptr) {

		grid_context **ctx = static_cast<grid_context **>(ptr);
		const grid_library_config config = grid_library_get_config();

		if (*ctx == nullptr)
				*ctx = new grid_context(config.backend);

		//if (((*ctx)->backend() == GRID_BACKEND_REF) || (config.validate))
				grid_ref_update_task_list(ntasks, nlevels, natoms, nkinds, nblocks, block_offsets, atom_positions,
																	atom_kinds, basis_sets, level_list, iatom_list, jatom_list, iset_list,
																	jset_list, ipgf_list, jpgf_list, border_mask_list, block_num_list,
																	radius_list, rab_list, (*ctx)->ref_backend_);

#ifdef __GRID_CUDA
		if (*ctx.backend() == GRID_BACKEND_GPU) {
				grid_gpu_update_task_list(ntasks, nlevels, natoms, nkinds, nblocks, block_offsets, atom_positions,
																	atom_kinds, basis_sets, level_list, iatom_list, jatom_list, iset_list,
																	jset_list, ipgf_list, jpgf_list, border_mask_list, block_num_list,
																	radius_list, rab_list, (*ctx).backend());
				return;
		}
#endif
		// update all informations for the tasks queues etc...

		(*ctx)->update_block_offsets(nblocks, block_offsets);
		(*ctx)->update_atoms_position(natoms, atom_positions);
		(*ctx)->update_atoms_kinds(natoms, atom_kinds);
		(*ctx)->update_basis_set(nkinds, basis_sets);
		(*ctx)->update_task_lists(nlevels, ntasks, level_list, iatom_list, jatom_list,
													 iset_list, jset_list, ipgf_list, jpgf_list,
													 border_mask_list, block_num_list, radius_list, rab_list);
		(*ctx)->update_grid(nlevels);

		// Find largest Cartesian subblock size.
		(*ctx)->maxco_ = 0;
		for (int i = 0; i < nkinds; i++) {
				(*ctx)->maxco_ = std::max((*ctx)->maxco_, (*ctx)->basis_sets_[i]->maxco);
		}

		if (config.apply_cutoff) {
				(*ctx)->apply_cutoff(true);
		}
}

/*******************************************************************************
 * \brief Deallocates given task list, basis_sets have to be freed separately.
 * \author Ole Schuett
 ******************************************************************************/
extern "C" void grid_free_task_list(void *ptr) {
		grid_context *ctx = static_cast<grid_context *>(ptr);
		delete ctx;
}

/*******************************************************************************
 * \brief Collocate all tasks of in given list onto given grids.
 *        See grid_task_list.h for details.
 ******************************************************************************/
extern "C" void grid_collocate_task_list(void *ptr, const bool orthorhombic,
																				 const enum grid_func func, const int nlevels,
																				 const int *npts_global, const int *npts_local,
																				 const int *shift_local, const int *border_width,
																				 const double *dh, const double *dh_inv,
																				 grid_buffer *pab_blocks, double **grid) {
		grid_context *ctx = static_cast<grid_context *>(ptr);
		if(ctx->backend() == GRID_BACKEND_REF) {
				grid_ref_collocate_task_list(ctx->backend_context(), orthorhombic, func, nlevels,
																		 npts_global, npts_local, shift_local,
																		 border_width, dh, dh_inv, pab_blocks, grid);
				return;
		}

#ifdef __GRID_CUDA
		if (ctx->backend() == GRID_BACKEND_GPU) {
				grid_gpu_collocate_task_list(ctx->backend_context(), orthorhombic, func, nlevels,
																		 npts_global, npts_local, shift_local,
																		 border_width, dh, dh_inv, pab_blocks, grid);
				return;
		}
#endif

		/* opnly the cpu backend uses this for now. The gpu backend will also use this afterwards */
		ctx->set_pab_blocks(pab_blocks);
		ctx->set_orthorhombic(orthorhombic);
		ctx->set_func(func);
		for (auto level = 0u; level < ctx->grid().size(); level++) {
				int local_size__[3] = {npts_local[3 * level + 2], npts_local[3 * level + 1], npts_local[3 * level]};
				int full_size__[3] = {npts_global[3 * level + 2], npts_global[3 * level + 1], npts_global[3 * level]};
				int shift_local__[3] = {shift_local[3 * level + 2], shift_local[3 * level + 1], shift_local[3 * level]};
				int border_width__[3] = {border_width[3 * level + 2], border_width[3 * level + 1], border_width[3 * level]};
				ctx->grid(level).set_grid_parameters(orthorhombic,
																						 full_size__,
																						 local_size__,
																						 shift_local__,
																						 border_width__,
																						 &dh[9 * level],
																						 &dh_inv[9 * level],
																						 grid[level]);
				ctx->grid(level).zero();
		}

		ctx->collocate();

		const grid_library_config config = grid_library_get_config();
		if (config.validate) {
				collocate_test(ctx->ref_backend_, orthorhombic, func, nlevels,
											 npts_global, npts_local, shift_local,
											 border_width, dh, dh_inv, pab_blocks, grid);
		}
}


/*******************************************************************************
 * \brief Integrate all tasks of in given list from given grids.
 *        See grid_task_list.h for details.
 ******************************************************************************/
extern "C" void grid_integrate_task_list(
		void *ptr, const bool orthorhombic, const bool compute_tau,
		const int natoms, const int nlevels, const int *npts_global,
		const int *npts_local, const int *shift_local,
		const int *border_width, const double *dh,
		const double *dh_inv, grid_buffer *const pab_blocks,
		double **grid, grid_buffer *hab_blocks, double *forces,
		double *virial) {

		assert(ptr != nullptr);
		grid_context *ctx = static_cast<grid_context *>(ptr);
		assert(forces == NULL || pab_blocks != NULL);
		assert(virial == NULL || pab_blocks != NULL);

		if (ctx->backend() == GRID_BACKEND_REF)
		{
				grid_ref_integrate_task_list(ctx->ref_backend_, orthorhombic, compute_tau,
																		 natoms, nlevels, npts_global, npts_local,
																		 shift_local, border_width, dh, dh_inv,
																		 pab_blocks, (const double **)grid, hab_blocks, forces, virial);
				return;
		}

#ifdef __GRID_CUDA
		if (ctx->backend() == GRID_BACKEND_GPU) {
				grid_gpu_integrate_task_list(ctx->backend_context(), orthorhombic, compute_tau,
																		 natoms, nlevels, npts_global, npts_local,
																		 shift_local, border_width, dh, dh_inv,
																		 pab_blocks, (const double **)grid, hab_blocks, forces, virial);
				return;
		}
#endif

		ctx->set_orthorhombic(orthorhombic);

		/* there are no allocation here. We only use the pointers internally */
		ctx->set_pab_blocks(pab_blocks);
		ctx->set_hab_blocks(hab_blocks);

		//#pragma omp parallel for
		for (auto level = 0u; level < ctx->grid().size(); level++) {
				int local_size__[3] = {npts_local[3 * level + 2], npts_local[3 * level + 1], npts_local[3 * level]};
				int full_size__[3] = {npts_global[3 * level + 2], npts_global[3 * level + 1], npts_global[3 * level]};
				int shift_local__[3] = {shift_local[3 * level + 2], shift_local[3 * level + 1], shift_local[3 * level]};
				int border_width__[3] = {border_width[3 * level + 2], border_width[3 * level + 1], border_width[3 * level]};
				ctx->grid(level).set_grid_parameters(orthorhombic,
																						 full_size__,
																						 local_size__,
																						 shift_local__,
																						 border_width__,
																						 &dh[9 * level],
																						 &dh_inv[9 * level],
																						 grid[level]);
		}

		bool calculate_virial = (virial != NULL);
		bool calculate_forces = (forces != NULL);

		ctx->calculate_tau(compute_tau);
		ctx->calculate_forces(calculate_forces);
		ctx->calculate_virial(calculate_virial);

		if (calculate_forces) {
				ctx->forces().resize(natoms, 3);
				ctx->virial().resize(3, 3);
				ctx->forces().zero();
				ctx->virial().zero();
		}

	ctx->integrate();

	if (calculate_forces) {
			if (calculate_virial) {
					virial[0] = ctx->virial(0, 0);
					virial[1] = ctx->virial(0, 1);
					virial[2] = ctx->virial(0, 2);
					virial[3] = ctx->virial(1, 0);
					virial[4] = ctx->virial(1, 1);
					virial[5] = ctx->virial(1, 2);
					virial[6] = ctx->virial(2, 0);
					virial[7] = ctx->virial(2, 1);
					virial[8] = ctx->virial(2, 2);
			}

			memcpy(forces, ctx->forces().at(), sizeof(double) * ctx->forces().size());
			ctx->forces().clear();
			ctx->virial().clear();
	}

	const grid_library_config config = grid_library_get_config();
	if (config.validate) {
			integrate_test(ctx->ref_backend_, orthorhombic, compute_tau,
										 natoms, nlevels, npts_global, npts_local,
										 shift_local, border_width, dh, dh_inv,
										 pab_blocks, (const double **)grid, hab_blocks, forces, virial);
	}
}

static void collocate_test(void *ptr, const bool orthorhombic,
													 const enum grid_func func, const int nlevels,
													 const int *npts_global, const int *npts_local,
													 const int *shift_local, const int *border_width,
													 const double *dh, const double *dh_inv,
													 grid_buffer *pab_blocks, double **grid) {
		grid_context *ctx = static_cast<grid_context *>(ptr);

// Allocate space for reference results.
		double **grid_ref = (double **)malloc(sizeof(double*) * nlevels);
		int max_size = 0;

		for (int i = 0; i < nlevels; i++)
				max_size = std::max(max_size, npts_local[3 * i] * npts_local[3 * i + 1] * npts_local[3 * i + 2]);

		grid_ref[0] = (double *)malloc(sizeof(double) * max_size * nlevels);

		for (int i = 1 ; i < nlevels; i++)
				grid_ref[i] = grid_ref[i - 1] + max_size;

		// Call reference implementation.
		grid_ref_collocate_task_list(
				ctx->ref_backend_, orthorhombic, func, nlevels, npts_global, npts_local,
				shift_local, border_width, dh, dh_inv, pab_blocks, grid_ref);

		// Compare results.
		const double tolerance = 1e-12;
		double max_rel_diff = 0.0;

		for (int level = 0; level < nlevels; level++) {
				tensor1<double, 3> grid__(grid_ref[level], npts_local[3 * level + 2], npts_local[3 * level + 1], npts_local[3 * level]);
				tensor1<double, 3> grid___(grid[level], npts_local[3 * level + 2], npts_local[3 * level + 1], npts_local[3 * level]);
				for (int k = 0; k < npts_local[3 * level + 2]; k++) {
						for (int j = 0; j < npts_local[3 * level + 1]; j++) {
								for (int i = 0; i < npts_local[3 * level]; i++) {
										const double ref_value = grid__(k, j, i);
										const double diff = fabs(grid___(k, j, i) - ref_value);
										const double rel_diff = diff / std::max(1.0, fabs(ref_value));
										max_rel_diff = std::max(max_rel_diff, rel_diff);
										if (rel_diff > tolerance) {
												fprintf(stderr, "Error: Validation failure in grid collocate\n");
												fprintf(stderr, "   diff:     %le\n", diff);
												fprintf(stderr, "   rel_diff: %le\n", rel_diff);
												fprintf(stderr, "   value:    %le\n", ref_value);
												fprintf(stderr, "   level:    %i\n", level);
												fprintf(stderr, "   ijk:      %i  %i  %i\n", i, j, k);
												abort();
										}
								}
						}
				}
				printf("Validated grid collocate, max rel. diff: %le\n", max_rel_diff);
		}
		free(grid_ref[0]);
		free(grid_ref);
}

static void integrate_test(	void *ptr, const bool orthorhombic, const bool compute_tau,
														const int natoms, const int nlevels, const int *npts_global,
														const int *npts_local, const int *shift_local,
														const int *border_width, const double *dh,
														const double *dh_inv, grid_buffer *const pab_blocks,
														const double **grid, grid_buffer *hab_blocks, double *forces,
														double *virial) {
// Allocate space for reference results.
		const int hab_length = hab_blocks->size / sizeof(double);
		grid_buffer *hab_blocks_ref = NULL;
		grid_create_buffer(hab_length, &hab_blocks_ref);
		tensor1<double, 2> forces_ref(natoms, 3), virial_ref(3, 3);

		// Call reference implementation.
		grid_ref_integrate_task_list(
				ptr, orthorhombic, compute_tau, natoms, nlevels, npts_global,
				npts_local, shift_local, border_width, dh, dh_inv, pab_blocks, grid,
				hab_blocks_ref, (forces != NULL) ? forces_ref.at() : NULL,
				(virial != NULL) ? virial_ref.at() : NULL);

		// Compare hab.
		const double hab_tolerance = 1e-12;
		double hab_max_rel_diff = 0.0;
		for (int i = 0; i < hab_length; i++) {
				const double ref_value = hab_blocks_ref->host_buffer[i];
				const double test_value = hab_blocks->host_buffer[i];
				const double diff = std::abs(test_value - ref_value);
				const double rel_diff = diff / std::max(1.0, fabs(ref_value));
				hab_max_rel_diff = fmax(hab_max_rel_diff, rel_diff);
				if (rel_diff > hab_tolerance) {
						fprintf(stderr, "Error: Validation failure in grid integrate\n");
						fprintf(stderr, "   hab diff:     %le\n", diff);
						fprintf(stderr, "   hab rel_diff: %le\n", rel_diff);
						fprintf(stderr, "   hab value:    %le\n", ref_value);
						fprintf(stderr, "   hab i:        %i\n", i);
						abort();
				}
		}

		// Compare forces.
		const double forces_tolerance = 1e-8; // account for higher numeric noise
		double forces_max_rel_diff = 0.0;
		if (forces != NULL) {
				tensor1<double, 2> forces_(forces, natoms, 3);
				for (int iatom = 0; iatom < natoms; iatom++) {
						for (int idir = 0; idir < 3; idir++) {
								const double ref_value = forces_ref(iatom, idir);
								const double test_value = forces_(iatom, idir);
								const double diff = std::abs(test_value - ref_value);
								const double rel_diff = diff / fmax(1.0, fabs(ref_value));
								forces_max_rel_diff = std::max(forces_max_rel_diff, rel_diff);
								if (rel_diff > forces_tolerance) {
										fprintf(stderr, "Error: Validation failure in grid integrate\n");
										fprintf(stderr, "   forces diff:     %le\n", diff);
										fprintf(stderr, "   forces rel_diff: %le\n", rel_diff);
										fprintf(stderr, "   forces value:    %le\n", ref_value);
										fprintf(stderr, "   forces atom:     %i\n", iatom);
										fprintf(stderr, "   forces dir:      %i\n", idir);
										abort();
								}
						}
				}
		}

		// Compare virial.
		const double virial_tolerance = 1e-8; // account for higher numeric noise
		double virial_max_rel_diff = 0.0;
		if (virial != NULL) {
				tensor1<double, 2> virial_(virial, 3, 3);
				for (int i = 0; i < 3; i++) {
						for (int j = 0; j < 3; j++) {
								const double ref_value = virial_ref(i, j);
								const double test_value = virial_(i, j);
								const double diff = std::abs(test_value - ref_value);
								const double rel_diff = diff / fmax(1.0, fabs(ref_value));
								virial_max_rel_diff = std::max(virial_max_rel_diff, rel_diff);
								if (rel_diff > virial_tolerance) {
										fprintf(stderr, "Error: Validation failure in grid integrate\n");
										fprintf(stderr, "   virial diff:     %le\n", diff);
										fprintf(stderr, "   virial rel_diff: %le\n", rel_diff);
										fprintf(stderr, "   virial value:    %le\n", ref_value);
										fprintf(stderr, "   virial ij:       %i  %i\n", i, j);
										abort();
								}
						}
				}
		}

		printf("Validated grid_integrate, max rel. diff: %le %le %le\n",
				 hab_max_rel_diff, forces_max_rel_diff, virial_max_rel_diff);
		grid_free_buffer(hab_blocks_ref);
}
