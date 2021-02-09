/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#ifdef __GRID_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#endif

extern "C" {
#include "../common/grid_common.h"
}

#include "grid_info.hpp"
#include "utils.hpp"
#include "cpu_handler.hpp"

cpu_handler::cpu_handler() {
		initialize(1);
}

void cpu_handler::initialize(const int i) {
		this->Exp_.resize(3, 16, 16);
		this->alpha_.resize(3, 10, 10, 10);
		this->pol_.resize(3, 10, 32);
		this->cube_.resize(16, 16, 16);
		this->coef_.resize(16, 16, 16);

		this->scratch = (double *)malloc(sizeof(double) * 32768);
		// aligned_alloc(sysconf(_SC_PAGESIZE), sizeof(double) * 32768);
		this->scratch_size_ = 32768;
		this->device_id.clear();
		this->device_id.resize(1);

		/* to suppress when we remove the spherical cutoff */
		this->cmax_ = 512 * 3;
		this->map_.resize(3, this->cmax_);
}

cpu_handler::~cpu_handler() {
		this->grid_.clear();
		device_id.clear();
		this->cube_.clear();
		this->coef_.clear();
		this->Exp_.clear();
		this->alpha_.clear();
		this->pol_.clear();
		this->map_.clear();
		this->hmatgridp.clear();
		this->coef_ijk.clear();
		this->cube_tmp.clear();
		this->pab_.clear();
		this->pab_prep_.clear();
		this->work_.clear();
		free(this->scratch);
		this->scratch = nullptr;
}

void cpu_handler::initialize_basis_vectors(const double dh[3][3], const double dh_inv[3][3]) {
		this->dh[0][0] = dh[0][0];
		this->dh[0][1] = dh[0][1];
		this->dh[0][2] = dh[0][2];
		this->dh[1][0] = dh[1][0];
		this->dh[1][1] = dh[1][1];
		this->dh[1][2] = dh[1][2];
		this->dh[2][0] = dh[2][0];
		this->dh[2][1] = dh[2][1];
		this->dh[2][2] = dh[2][2];

		this->dh_inv[0][0] = dh_inv[0][0];
		this->dh_inv[0][1] = dh_inv[0][1];
		this->dh_inv[0][2] = dh_inv[0][2];
		this->dh_inv[1][0] = dh_inv[1][0];
		this->dh_inv[1][1] = dh_inv[1][1];
		this->dh_inv[1][2] = dh_inv[1][2];
		this->dh_inv[2][0] = dh_inv[2][0];
		this->dh_inv[2][1] = dh_inv[2][1];
		this->dh_inv[2][2] = dh_inv[2][2];

		/* Only used when we are in the non  orthorombic case */
		this->dx[2] = this->dh[0][0] * this->dh[0][0] +
				this->dh[0][1] * this->dh[0][1] +
				this->dh[0][2] * this->dh[0][2];
		this->dx[1] = this->dh[1][0] * this->dh[1][0] +
				this->dh[1][1] * this->dh[1][1] +
				this->dh[1][2] * this->dh[1][2];
		this->dx[0] = this->dh[2][0] * this->dh[2][0] +
				this->dh[2][1] * this->dh[2][1] +
				this->dh[2][2] * this->dh[2][2];
}
