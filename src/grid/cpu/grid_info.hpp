/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifndef GRID_INFO_HPP
#define GRID_INFO_HPP

#include <cmath>
#include <cstring>
#include <vector>
#include "../common/tensor.hpp"

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

class grid_info {
private:
    tensor1<double, 3> grid_;
    int full_size_[3] = {0, 0, 0};    /* size of the global grid */
public:
    int window_shift_[3] = {0, 0, 0}; /* lower corner of the window. Should be between lower
                          * corner and upper corner of the local grid */
    int window_size_[3]  = {0, 0, 0};  /* size of the window where computations should be
                          * done */
    int lower_corner_[3] = {0, 0, 0}; /* coordinates of the lower corner of the local part of
                          * the grid. It can be different from the window where
                          * computations should be done. The upper corner can be
                          * deduced with the sum of the grid size and the lower
                          * corner */
    /* only relevant when the tensor represents a grid */
    double dh[3][3];
    double dh_inv[3][3];
    bool orthogonal[3] = {false, false, false};

    grid_info(grid_info const& src) = default;
    grid_info&
    operator=(grid_info const& src) = default;

    inline grid_info&operator=(grid_info &&src) {
        if (this != &src) {
            grid_ = src.grid_;
            memcpy(window_shift_, src.window_shift_, sizeof(int) * 3);
            memcpy(window_size_, src.window_size_, sizeof(int) * 3);
            memcpy(full_size_, src.full_size_, sizeof(int) * 3);
            memcpy(&dh[0][0], &src.dh[0][0], sizeof(double) * 9);
            memcpy(&dh_inv[0][0], &src.dh_inv[0][0], sizeof(double) * 9);
            memcpy(lower_corner_, src.lower_corner_, sizeof(int) * 3);
            memcpy(orthogonal, src.orthogonal, sizeof(bool) * 3);
        }
        return *this;
    }

    inline grid_info(grid_info &&src) {
        grid_ = src.grid_;
        memcpy(window_shift_, src.window_shift_, sizeof(int) * 3);
        memcpy(window_size_, src.window_size_, sizeof(int) * 3);
        memcpy(full_size_, src.full_size_, sizeof(int) * 3);
        memcpy(&dh[0][0], &src.dh[0][0], sizeof(double) * 9);
        memcpy(&dh_inv[0][0], &src.dh_inv[0][0], sizeof(double) * 9);
        memcpy(lower_corner_, src.lower_corner_, sizeof(int) * 3);
        memcpy(orthogonal, src.orthogonal, sizeof(bool) * 3);
    }

    grid_info() {
    };

    ~grid_info() {
        grid_.clear();
    }

    grid_info(const int *local_size, const int *full_size) {
        resize(local_size, full_size);
    }

    grid_info(double *ptr, const int *local_size, const int *full_size) {
        grid_.update_pointer(ptr);
        resize(local_size, full_size);
    }

    const bool is_distributed() const {
        return ((grid_.size(0) != full_size_[0]) ||
                (grid_.size(1) != full_size_[1]) ||
                (grid_.size(2) != full_size_[2]));
    }

    void zero() {
        grid_.zero();
    }
    void clear() {
        grid_.clear();
    }

    void update_pointer(double *ptr) {
        grid_.update_pointer(ptr);
    }

    int size() const {
        return grid_.size();
    }
    inline double *at() {
        return grid_.at();
    }

    inline double *at(const int x, const int y, const int z) {
        return grid_.at(x, y, z);
    }

    inline double &operator()(int x, int y, int z) {
        return grid_(x, y, z);
    }

    inline const double operator()(int x, int y, int z) const {
        return grid_(x, y, z);
    }

    inline int full_size(const int i) const {
        assert((i >= 0) && (i < 3));
        return full_size_[i];
    }

    inline int size(const int i) const {
        assert((i >= 0) && (i < 3));
        return grid_.size(i);
    }

    inline int window_size(const int i) const {
        assert((i >= 0) && (i < 3));
        return window_size_[i];
    }

    inline int window_shift(const int i) const {
        assert((i >= 0) && (i < 3));
        return window_shift_[i];
    }

    inline int lower_corner(const int i) const {
        assert((i >= 0) && (i < 3));
        return lower_corner_[i];
    }

    void set_full_size(const int *full_size) {
        full_size_[0] = full_size[0];
        full_size_[1] = full_size[1];
        full_size_[2] = full_size[2];
    }

    void resize(const int *local_size, const int *full_size) {
        grid_.resize(local_size[0], local_size[1], local_size[2]);
        set_full_size(full_size);
    }

    void resize(double *ptr, const int *local_size, const int *full_size) {
        grid_.update_pointer(ptr);
        grid_.resize(local_size[0], local_size[1], local_size[2]);
        set_full_size(full_size);
    }


    inline void setup_grid_window(const int *const shift_local,
                                  const int *const border_width,
                                  const int border_mask) {
        for (int d = 0; d < 3; d++) {
            this->lower_corner_[d] = shift_local[d];
            this->window_shift_[d] = 0;
            this->window_size_[d] = this->grid_.size(d);
            if (this->grid_.size(d) != this->full_size_[d]) {
                this->window_size_[d]--;
            }
        }

        if (border_width) {
            if (border_mask & (1 << 0))
                this->window_shift_[2] += border_width[2];
            if (border_mask & (1 << 1))
                this->window_size_[2] -= border_width[2];
            if (border_mask & (1 << 2))
                this->window_shift_[1] += border_width[1];
            if (border_mask & (1 << 3))
                this->window_size_[1] -= border_width[1];
            if (border_mask & (1 << 4))
                this->window_shift_[0] += border_width[0];
            if (border_mask & (1 << 5))
                this->window_size_[0] -= border_width[0];
        }
    }

    void set_grid_parameters(
        const bool orthorhombic,
        const int grid_full_size[3],  /* size of the full grid */
        const int grid_local_size[3], /* size of the local grid block */
        const int shift_local[3],     /* coordinates of the lower coordinates of the
                                         local grid window */
        const int border_width[3],    /* width of the borders */
        const double
        *dh, /* displacement vectors of the grid (cartesian) -> (ijk) */
        const double *dh_inv, /* (ijk) -> (x,y,z) */
        double *grid__) {
        grid_.update_pointer(grid__);
        resize(grid_local_size, grid_full_size);

        /* the grid is divided over several ranks or not periodic */
        if ((grid_.size(0) != grid_full_size[0]) ||
            (grid_.size(1) != grid_full_size[1]) ||
            (grid_.size(2) != grid_full_size[2])) {
            setup_grid_window(shift_local, border_width, 0);
        } else {
            this->window_shift_[0] = 0;
            this->window_shift_[1] = 0;
            this->window_shift_[2] = 0;

            this->window_size_[0] = this->grid_.size(0);
            this->window_size_[1] = this->grid_.size(1);
            this->window_size_[2] = this->grid_.size(2);
        }

        this->dh[0][0] = dh[0];
        this->dh[0][1] = dh[1];
        this->dh[0][2] = dh[2];
        this->dh[1][0] = dh[3];
        this->dh[1][1] = dh[4];
        this->dh[1][2] = dh[5];
        this->dh[2][0] = dh[6];
        this->dh[2][1] = dh[7];
        this->dh[2][2] = dh[8];

        this->dh_inv[0][0] = dh_inv[0];
        this->dh_inv[0][1] = dh_inv[1];
        this->dh_inv[0][2] = dh_inv[2];
        this->dh_inv[1][0] = dh_inv[3];
        this->dh_inv[1][1] = dh_inv[4];
        this->dh_inv[1][2] = dh_inv[5];
        this->dh_inv[2][0] = dh_inv[6];
        this->dh_inv[2][1] = dh_inv[7];
        this->dh_inv[2][2] = dh_inv[8];

        verify_orthogonality();

        if (orthorhombic) {
            this->orthogonal[0] = true;
            this->orthogonal[1] = true;
            this->orthogonal[2] = true;
        }
    }

    inline void verify_orthogonality() {
        double norm1, norm2, norm3;

        norm1 = dh[0][0] * dh[0][0] + dh[0][1] * dh[0][1] + dh[0][2] * dh[0][2];
        norm2 = dh[1][0] * dh[1][0] + dh[1][1] * dh[1][1] + dh[1][2] * dh[1][2];
        norm3 = dh[2][0] * dh[2][0] + dh[2][1] * dh[2][1] + dh[2][2] * dh[2][2];

        norm1 = 1.0 / sqrt(norm1);
        norm2 = 1.0 / sqrt(norm2);
        norm3 = 1.0 / sqrt(norm3);

        /* x z */
        orthogonal[0] =
            ((std::fabs(dh[0][0] * dh[2][0] + dh[0][1] * dh[2][1] + dh[0][2] * dh[2][2]) *
              norm1 * norm3) < 1e-12);
        /* y z */
        orthogonal[1] =
            ((std::fabs(dh[1][0] * dh[2][0] + dh[1][1] * dh[2][1] + dh[1][2] * dh[2][2]) *
              norm2 * norm3) < 1e-12);
        /* x y */
        orthogonal[2] =
            ((std::fabs(dh[0][0] * dh[1][0] + dh[0][1] * dh[1][1] + dh[0][2] * dh[1][2]) *
              norm1 * norm2) < 1e-12);
    }

    inline void extract_sub_grid(const int *lower_corner, const int *upper_corner,
                      const int *position, tensor1<double, 3> &subgrid) {
        int position1[3] = {0, 0, 0};

        if (position) {
            position1[0] = position[0];
            position1[1] = position[1];
            position1[2] = position[2];
        }

        const int sizex = upper_corner[2] - lower_corner[2];
        const int sizey = upper_corner[1] - lower_corner[1];
        const int sizez = upper_corner[0] - lower_corner[0];

        for (int z = 0; z < sizez; z++) {
            /* maybe use matcopy from libxsmm if possible */
            double *__restrict__ src =
                grid_.at(lower_corner[0] + z - this->window_shift_[0],
                         lower_corner[1] - this->window_shift_[1],
                         lower_corner[2] - this->window_shift_[2]);
            double *__restrict__ dst = subgrid.at(position1[0] + z, position1[1], position1[2]);
            for (int y = 0; y < sizey; y++) {
                memcpy(dst, src, sizex * sizeof(double));
// #ifdef __LIBXSMM
//                 LIBXSMM_PRAGMA_SIMD
// #else
// #pragma GCC ivdep
// #endif
//                     for (int x = 0; x < sizex; x++) {
//                         dst[x] = src[x];
//                     }

                 dst += subgrid_.ld();
                 src += grid_.ld();
            }
        }
        return;
    }

    double integrate() {
        double sum = 0.0;

        const double *__restrict__ src = grid_.at();
        for (int i = 0; i < size(); i++)
            sum += src[i];
        return sum;
    }

    void add_sub_grid(const int *lower_corner, const int *upper_corner,
                      const int *position, const tensor1<double, 3> &subgrid) {
        int position1[3] = {0, 0, 0};

        if (position) {
            position1[0] = position[0];
            position1[1] = position[1];
            position1[2] = position[2];
        }

        const int sizex = upper_corner[2] - lower_corner[2];
        const int sizey = upper_corner[1] - lower_corner[1];
        const int sizez = upper_corner[0] - lower_corner[0];

        for (int z = 0; z < sizez; z++) {
            double *__restrict__ dst =
                grid_.at(lower_corner[0] + z, lower_corner[1], lower_corner[2]);
            const double *__restrict__ src =
                subgrid.at(position1[0] + z, position1[1], position1[2]);
            for (int y = 0; y < sizey; y++) {
#ifdef __LIBXSMM
                LIBXSMM_PRAGMA_SIMD
#else
#pragma GCC ivdep
#endif
                    for (int x = 0; x < sizex; x++) {
                        dst[x] += src[x];
                    }

                dst += grid_.ld();
                src += subgrid.ld();
            }
        }
        return;
    };
};
#endif
