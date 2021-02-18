#include <algorithm>
#include <iostream>

extern "C" {
#include "common/grid_constants.h"
#include "common/grid_library.h"
#include "ref/grid_ref_task_list.h"
}

#include "common/task.hpp"
#include "common/grid_info.hpp"

#include "cpu/grid_context_cpu.hpp"

void grid_context::update_backend(const grid_backend backend__)
{
		backend_ = backend__;
		if (ref_backend_ == nullptr)
				ref_backend_ = grid_ref_create_task_list();

		if (!ref_backend_) {
				fprintf(stderr, "resource allocation failed\n");
				abort();
		}
		switch(backend__) {
		case GRID_BACKEND_REF:
				backend_ctx_ = ref_backend_;
				break;
#ifdef __GRID_CUDA
		case GRID_BACKEND_GPU: {
				if (backend_ctx_ == nullptr) {
						backend_ctx_ = new gpu_backend(*this);
				}
		}
				break;
#endif
		case GRID_BACKEND_CPU:
				if (backend_ctx_ == nullptr) {
						backend_ctx_ = new cpu_backend(*this);
				}
				break;
		default:
				break;
		}
}

void grid_context::change_backend(const grid_backend backend__) {
		switch(backend_) {
		case GRID_BACKEND_REF:
				break;
#ifdef __GRID_CUDA
		case GRID_BACKEND_GPU:
				delete static_cast<gpu_backend*>(backend_ctx_);
				break;
#endif
		case GRID_BACKEND_CPU:
				delete static_cast<cpu_backend*>(backend_ctx_);
				break;
		default:
				break;
		}
		update_backend(backend__);
}

grid_context::grid_context(const grid_backend backend) {
		if (backend == GRID_BACKEND_AUTO) {
#ifdef __GRID_CUDA
				update_backend(GRID_BACKEND_GPU);
#else
				update_backend(GRID_BACKEND_REF);
#endif
				return;
		}

		update_backend(backend);
}

grid_context::~grid_context() {
		block_offsets_.clear();
		atom_positions_.clear();
		atom_kinds_.clear();
		basis_sets_.clear();
		tasks_list_.clear();
		tasks_per_level_.clear();
		queues_.clear();
		grid_.clear();
		forces_.clear();
		virial_.clear();

		switch(backend_) {
#ifdef __GRID_CUDA
		case GRID_BACKEND_GPU:
				delete static_cast<gpu_backend*>(backend_ctx_);
				break;
#endif
		case GRID_BACKEND_CPU:
				delete static_cast<cpu_backend*>(backend_ctx_);
				break;
		default:
				break;
		}

		grid_ref_free_task_list(ref_backend_);
}

// void grid_context::return_dh(const int level, double *const dh) {
//	dh[0] = this->grid_[level].dh[0][0];
//	dh[1] = this->grid_[level].dh[0][1];
//	dh[2] = this->grid_[level].dh[0][2];
//	dh[3] = this->grid_[level].dh[1][0];
//	dh[4] = this->grid_[level].dh[1][1];
//	dh[5] = this->grid_[level].dh[1][2];
//	dh[6] = this->grid_[level].dh[2][0];
//	dh[7] = this->grid_[level].dh[2][1];
//	dh[8] = this->grid_[level].dh[2][2];
// }

// void grid_context::return_dh_inv(const int level, double *const dh_inv) {
//	dh_inv[0] = this->grid_[level].dh_inv[0][0];
//	dh_inv[1] = this->grid_[level].dh_inv[0][1];
//	dh_inv[2] = this->grid_[level].dh_inv[0][2];
//	dh_inv[3] = this->grid_[level].dh_inv[1][0];
//	dh_inv[4] = this->grid_[level].dh_inv[1][1];
//	dh_inv[5] = this->grid_[level].dh_inv[1][2];
//	dh_inv[6] = this->grid_[level].dh_inv[2][0];
//	dh_inv[7] = this->grid_[level].dh_inv[2][1];
//	dh_inv[8] = this->grid_[level].dh_inv[2][2];
// }


void grid_context::update_queue_length(const int queue_length) {
	this->queue_length_ = queue_length;
}

void grid_context::update_atoms_position(const int natoms,
																				 const double *atoms_positions) {
	this->atom_positions_.clear();
	this->atom_positions_.resize(3 * natoms);

	for (auto i = 0u; i < this->atom_positions_.size(); i++) {
			this->atom_positions_[i] = atoms_positions[i];
	}
}

void grid_context::update_atoms_kinds(const int natoms, const int *atoms_kinds) {
	this->atom_kinds_.clear();
	this->atom_kinds_.resize(natoms);

	memcpy(&this->atom_kinds_[0], atoms_kinds, sizeof(int) * natoms);

	for (auto i = 0u; i < this->atom_kinds_.size(); i++) {
		this->atom_kinds_[i] -= 1;
	}
}

void grid_context::update_block_offsets(const int nblocks, const int *const block_offsets) {
	if (nblocks == 0)
		return;

	this->block_offsets_.clear();
	this->block_offsets_.resize(nblocks);

	memcpy(&this->block_offsets_[0], block_offsets, nblocks * sizeof(int));
}

void grid_context::update_basis_set(const int nkinds, const grid_basis_set **const basis_sets) {
	this->basis_sets_.clear();
	this->basis_sets_.resize(nkinds);
	memcpy(&this->basis_sets_[0], basis_sets, nkinds * sizeof(grid_basis_set *));
}

void grid_context::update_task_lists(const int nlevels, const int ntasks,
																		 const int *const level_list, const int *const iatom_list,
																		 const int *const jatom_list, const int *const iset_list,
																		 const int *const jset_list, const int *const ipgf_list,
																		 const int *const jpgf_list,
																		 const int *const border_mask_list,
																		 const int *block_num_list,
																		 const double *const radius_list,
																		 const double *rab_list) {
		if (nlevels == 0)
				return;

		// Count tasks per level.
		this->tasks_per_level_.clear();
		this->tasks_per_level_.resize(nlevels);
		this->tasks_list_.clear();
		this->tasks_list_.resize(ntasks);
		memset(&this->tasks_list_[0], 0, sizeof(task_info) * this->tasks_list_.size());
		this->queues_.clear();
		this->queues_.resize(nlevels);

		memset(&this->tasks_per_level_[0], 0, sizeof(int) * nlevels);
		for (int i = 0; i < ntasks; i++) {
				this->tasks_per_level_[level_list[i] - 1]++;
				assert(i == 0 || level_list[i] >= level_list[i - 1]); // expect ordered list
		}

		this->queues_[0] = &this->tasks_list_[0];

		for (auto i = 1u; i < this->tasks_per_level_.size(); i++) {
				this->queues_[i] = this->queues_[i - 1] + this->tasks_per_level_[i - 1];
		}

		int prev_block_num = -1;
		int prev_iset = -1;
		int prev_jset = -1;
		int prev_level = -1;
		for (int i = 0; i < ntasks; i++) {
				auto &task_ = this->tasks_list_[i];
				if (prev_level != (level_list[i] - 1)) {
						prev_level = level_list[i] - 1;
						prev_block_num = -1;
						prev_iset = -1;
						prev_jset = -1;
				}
				task_.level = level_list[i] - 1;
				task_.iatom = iatom_list[i] - 1;
				task_.jatom = jatom_list[i] - 1;
				task_.iset = iset_list[i] - 1;
				task_.jset = jset_list[i] - 1;
				task_.ipgf = ipgf_list[i] - 1;
				task_.jpgf = jpgf_list[i] - 1;
				task_.border_mask = border_mask_list[i];
				task_.block_num = block_num_list[i] - 1;
				task_.radius = radius_list[i];
				task_.rab[0] = rab_list[3 * i];
				task_.rab[1] = rab_list[3 * i + 1];
				task_.rab[2] = rab_list[3 * i + 2];
				task_.ikind = this->atom_kinds_[task_.iatom];
				task_.jkind = this->atom_kinds_[task_.jatom];
				const int iset = task_.iset;
				const int jset = task_.jset;
				const int ipgf = task_.ipgf;
				const int jpgf = task_.jpgf;
				const grid_basis_set *ibasis = this->basis_sets_[task_.ikind];
				const grid_basis_set *jbasis = this->basis_sets_[task_.jkind];
				const int ncoseta = ncoset(ibasis->lmax[iset]);
				const int ncosetb = ncoset(jbasis->lmax[jset]);

				task_.zeta[0] = ibasis->zet[iset * ibasis->maxpgf + ipgf];
				task_.zeta[1] = jbasis->zet[jset * jbasis->maxpgf + jpgf];

				const double *ra = &this->atom_positions_[3 * task_.iatom];
				const double zetp = task_.zeta[0] + task_.zeta[1];
				const double f = task_.zeta[1] / zetp;
				const double rab2 = task_.rab[0] * task_.rab[0] +
						task_.rab[1] * task_.rab[1] +
						task_.rab[2] * task_.rab[2];

				task_.prefactor = exp(-task_.zeta[0] * f * rab2);
				task_.zetp = zetp;

				const int block_num = task_.block_num;

				for (int i = 0; i < 3; i++) {
						task_.ra[i] = ra[i];
						task_.rp[i] = ra[i] + f * task_.rab[i];
						task_.rb[i] = ra[i] + task_.rab[i];
				}

				task_.lmax[0] = ibasis->lmax[iset];
				task_.lmax[1] = jbasis->lmax[jset];
				task_.lmin[0] = ibasis->lmin[iset];
				task_.lmin[1] = jbasis->lmin[jset];

				if ((block_num != prev_block_num) || (iset != prev_iset) ||
						(jset != prev_jset)) {
						task_.update_block_ = true;
						prev_block_num = block_num;
						prev_iset = iset;
						prev_jset = jset;
				} else {
						task_.update_block_ = false;
				}

				task_.offset[0] = ipgf * ncoseta;
				task_.offset[1] = jpgf * ncosetb;
		}

		// Find largest Cartesian subblock size.
		this->maxco_ = 0;
		for (auto &kind: this->basis_sets_) {
				this->maxco_ = std::max(this->maxco_, kind->maxco);
		}
}

void grid_context::update_grid(const int nlevels) {
	if (nlevels == 0)
		return;
	this->grid_.clear();
	this->grid_.resize(nlevels);
}

void grid_context::collocate() {
		switch(this->backend_) {
		case GRID_BACKEND_CPU: {
				cpu_backend *cpu_ctx = static_cast<cpu_backend *>(this->backend_ctx_);
				cpu_ctx->collocate();
		}
				break;
#ifdef GRID_CUDA
		case GRID_BACKEND_GPU: {
				gpu_backend *gpu_ctx = static_cast<gpu_backend *>(this->backend_ctx_);
				gpu_ctx->collocate();
		}
				break;
#else
		case GRID_BACKEND_GPU:
#endif
		case GRID_BACKEND_REF:
		case GRID_BACKEND_AUTO:
		default:
				break;
		}
}

void grid_context::integrate() {
		switch(this->backend_) {
		case GRID_BACKEND_CPU: {
				cpu_backend *cpu_ctx = static_cast<cpu_backend *>(this->backend_ctx_);
				cpu_ctx->integrate();
		}
				break;
#ifdef GRID_CUDA
		case GRID_BACKEND_GPU:{
				gpu_backend *gpu_ctx = static_cast<gpu_backend *>(this->backend_ctx_);
				gpu_ctx->integrate();
		}
#else
		case GRID_BACKEND_GPU:
#endif
		case GRID_BACKEND_REF:
		case GRID_BACKEND_AUTO:
		default:
				break;
		}
}
