/*----------------------------------------------------------------------------*/
/*  CP2K: A general program to perform molecular dynamics simulations         */
/*  Copyright 2000-2021 CP2K developers group <https://cp2k.org>              */
/*                                                                            */
/*  SPDX-License-Identifier: GPL-2.0-or-later                                 */
/*----------------------------------------------------------------------------*/

#ifndef CPU_HANDLER_HPP
#define CPU_HANDLER_HPP
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#ifdef __GRID_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#endif

extern "C" {
#include "../common/grid_constants.h"
}

#include "../common/Interval.hpp"
#include "../common/tensor.hpp"
#include "../common/task.hpp"
#include "../common/grid_info.hpp"
#include "utils.hpp"

class cpu_handler {
private:
    tensor1<double, 3> hmatgridp;
    tensor1<double, 3> coef_ijk;
    tensor1<double, 2> cube_tmp;
    /* number of compute device */
    std::vector<int> device_id;

    /* number of gaussians block in each list */
    int number_of_gaussian;

  /* some scratch storage to avoid malloc / free all the time */
    tensor1<double, 4> alpha_;
    tensor1<double, 3> pol_;
    tensor1<double, 3> coef_;
    tensor1<double, 3> cube_;
    tensor1<double, 3> T;
    tensor1<double, 3> W;
    tensor1<double, 2> work_, pab_, pab_prep_;

    /* tensors for the grid to collocate or integrate */
    /* original grid */
    grid_info grid_;

    /* int period[3]; */
    /* int lb_grid[3]; */


    double dh[3][3];
    double dh_inv[3][3];
    double dx[3];

    /* Only allocated in sequential mode */
    tensor1<double, 3> Exp_;
    size_t scratch_size_;
    int lmax;
    /* for the spherical cutoff */
    tensor1<int, 2> map_;
    double *scratch;

    bool durty;
    bool integrate_;

    int cmax_{32};
    int cube_size_[3] = {0, 0, 0};
    int cube_center_[3] = {0, 0, 0};
    int lb_cube_[3] = {0, 0, 0};
    int ub_cube_[3] = {0, 0, 0};
    double disr_radius_{0.0};
    double roffset_[3] = {0.0, 0.0, 0.0};

    struct int3 {
        int x, xmin, xmax;
    };

    std::vector<int3> i_x_;
    std::vector<int3> i_y_;
    std::vector<int3> i_z_;
public:
    bool apply_cutoff;
    bool orthogonal[3];

    enum grid_func func;
    int lmin_diff[2];
    int lmax_diff[2];

    grid_info &grid() {
        return grid_;
    }

    tensor1<double, 2> &work() {
        return work_;
    }

    tensor1<double, 2> &pab() {
        return pab_;
    }

    tensor1<double, 2> &pab_prep() {
        return pab_prep_;
    }

    cpu_handler();
    ~cpu_handler();
    void initialize(const int num_dev);
    void initialize_basis_vectors(const double dh[3][3], const double dh_inv[3][3]);
    void collocate(const bool use_ortho,
                   const task_info &task);
    void integrate(const bool use_ortho,
                   const task_info &task, tensor1<double, 2> &vab);

    void prepare_alpha(const task_info &task, const int *lmax);

    inline tensor1<double, 4> &alpha() {
        return alpha_;
    }

    inline const tensor1<double, 4> &alpha() const {
        return alpha_;
    }

    inline tensor1<double, 3> &coef() {
        return coef_;
    }

    inline const tensor1<double, 3> &coef() const {
        return coef_;
    }

    void compute_coefficients(const int *lmin,
                              const int *lmax,
                              const int lp,
                              const double prefactor,
                              const tensor1<double, 2> &pab);
private:
    inline void set_map() {
        this->map_.resize(3, cmax_ + 2);
//        memset(this->map_.at(), 0xff, sizeof(int) * this->map_.size());
        for (int i = 0; i < 3; i++) {
            int *__restrict map__ = this->map_.at(i, 0);
            for (int ig = 0; ig < this->cube_.size(i); ig++) {
                const int tmp = (cube_center_[i] + lb_cube_[i] + ig -
                                 this->grid_.lower_corner(i)) % ((int)this->grid_.full_size(i));
                map__[ig] = ((tmp < 0) ? (tmp + this->grid_.full_size(i)) : (tmp));
            }
        }
    }

    inline int compute_next_boundaries(const int y1, const int y,
                                       const int grid_size,
                                       const int cube_size) {
        return y1 + std::min(cube_size - y, grid_size - y1);
    }


    inline void compute_interval(const int *const map, const int full_size, const int size,
                                 const int cube_size, const int x1, int *x,
                                 int *const lower_corner, int *const upper_corner,
                                 const Interval &window) {
        if (size == full_size) {
            /* we have the full grid in that direction */
            /* lower boundary is within the window */
            *lower_corner = x1;
            /* now compute the upper corner */
            /* needs to be as large as possible. basically I take [x1..
             * min(grid.full_size, cube_size - x)] */

            *upper_corner = compute_next_boundaries(x1, *x, full_size, cube_size);

            {
                Interval tz(*lower_corner, *upper_corner);
                Interval res = tz.intersection_interval(window);
                *lower_corner = res.xmin();
                *upper_corner = res.xmax();
            }
        } else {
            *lower_corner = x1;
            *upper_corner = x1 + 1;

            // the map is always increasing by 1 except when we cross the boundaries of
            // the grid and pbc are applied. Since we are only interested in by a
            // subwindow of the full table we check that the next point is inside the
            // window of interest and is also equal to the previous point + 1. The last
            // check is pointless in practice.

            for (int i = *x + 1; (i < cube_size) && (*upper_corner == map[i]) &&
                     window.is_point_in_interval(map[i]);
                 i++) {
                (*upper_corner)++;
            }
        }
    }

    inline void update_loop_index(const int global_grid_size, int x1,
                                  int *const x) {
        *x += global_grid_size - x1 - 1;
    }

    template<bool add=true> inline  void apply_spherical_cutoff_ortho() {
  // a mapping so that the ig corresponds to the right grid point
        set_map();

        const Interval zwindow = {.xmin = this->grid_.window_shift_[0],
            .xmax = this->grid_.window_size_[0]};
        const Interval ywindow = {.xmin = this->grid_.window_shift_[1],
            .xmax = this->grid_.window_size_[1]};
        const Interval xwindow = {.xmin = this->grid_.window_shift_[2],
            .xmax = this->grid_.window_size_[2]};

        for (int kg = 0; kg < (int)this->cube_.size(0); kg++) {
            const int k = map_(0, kg);
            const int kd =
                (2 * (kg + lb_cube_[0]) - 1) / 2; // distance from center in grid points
            const double kr = kd * this->dh[2][2]; // distance from center in a.u.
            const double kremain = disr_radius_ * disr_radius_ - kr * kr;
            if ((kremain >= 0.0) && zwindow.is_point_in_interval(k)) {

                const int jgmin = std::ceil(-1e-8 - std::sqrt(kremain) * this->dh_inv[1][1]);
                for (int jg = jgmin; jg <= (1 - jgmin); jg++) {
                    const int j = map_(1, jg - lb_cube_[1]);

                    const double jr = ((2 * jg - 1) >> 1) *
                        this->dh[1][1]; // distance from center in a.u.
                    const double jremain = kremain - jr * jr;

                    if ((jremain >= 0.0) && ywindow.is_point_in_interval(j)) {
                        const int xmin = std::ceil(-1e-8 - std::sqrt(jremain) * this->dh_inv[0][0]);
                        const int xmax = 1 - xmin;

                        for (int x = xmin - lb_cube_[2];
                             x < std::min(xmax - lb_cube_[2], (int)this->cube_.size(2)); x++) {
                            const int x1 = map_(2, x);

                            if (!xwindow.is_point_in_interval(x1))
                                continue;

                            int lower_corner[3] = {k, j, x1};
                            int upper_corner[3] = {k + 1, j + 1, x1 + 1};

                            compute_interval(map_.at(2, 0), this->grid_.full_size(2),
                                             this->grid_.size(2), this->cube_.size(2), x1,
                                             &x, lower_corner + 2, upper_corner + 2, xwindow);

                            if (upper_corner[2] - lower_corner[2]) {
                                const int position1[3] = {kg, jg - lb_cube_[1], x};

                                if (add == true) {
                                    double *__restrict__ dst = this->grid_.at(lower_corner[0],
                                                                              lower_corner[1],
                                                                              lower_corner[2]);
                                    const double *__restrict__ src = this->cube_.at(position1[0],
                                                                                    position1[1],
                                                                                    position1[2]);

                                    const int sizex = upper_corner[2] - lower_corner[2];
#pragma GCC ivdep
                                    for (int x = 0; x < sizex; x++) {
                                        dst[x] += src[x];
                                    }
                                } else {
                                    double *__restrict__ dst = this->cube_.at(position1[0],
                                                                              position1[1],
                                                                              position1[2]);

                                    const double *__restrict__ src = this->grid_.at(lower_corner[0],
                                                                                   lower_corner[1],
                                                                                   lower_corner[2]);

                                    const int sizex = upper_corner[2] - lower_corner[2];
#pragma GCC ivdep
                                    for (int x = 0; x < sizex; x++) {
                                        dst[x] = src[x];
                                    }
                                }

                                if (this->grid_.size(2) == this->grid_.full_size(2))
                                    update_loop_index(this->grid_.full_size(2), x1, &x);
                                else
                                    x += upper_corner[2] - lower_corner[2] - 1;
                            }
                        }
                    }
                }
            }
        }
    }

    template<bool add=true> inline void apply_spherical_cutoff_generic(const double radius) {

        const double a = this->dh[0][0] * this->dh[0][0] +
            this->dh[0][1] * this->dh[0][1] +
            this->dh[0][2] * this->dh[0][2];
        const double a_inv = 1.0 / a;

        set_map();

        const Interval zwindow = {.xmin = this->grid_.window_shift(0),
            .xmax = this->grid_.window_size(0)};
        const Interval ywindow = {.xmin = this->grid_.window_shift(1),
            .xmax = this->grid_.window_size(1)};
        const Interval xwindow = {.xmin = this->grid_.window_shift(2),
            .xmax = this->grid_.window_size(2)};

        for (auto k = 0; k < this->cube_.size(0); k++) {
            const int iz = map_(0, k);

            if (!zwindow.is_point_in_interval(iz))
                continue;

            const double tz = (k + lb_cube_[0] - roffset_[0]);
            const double z[3] = {tz * this->dh[2][0], tz * this->dh[2][1],
                tz * this->dh[2][2]};

            for (auto j = 0; j < this->cube_.size(1); j++) {
                const int iy = map_(1, j);

                if (!ywindow.is_point_in_interval(iy))
                    continue;

                const double ty = (j - roffset_[1] + lb_cube_[1]);
                const double y[3] = {z[0] + ty * this->dh[1][0],
                    z[1] + ty * this->dh[1][1],
                    z[2] + ty * this->dh[1][2]};

                /* Sqrt[(-2 x1 \[Alpha] - 2 y1 \[Beta] - 2 z1 \[Gamma])^2 - */
                /*                                            4 (x1^2 + y1^2 + z1^2)
                 * (\[Alpha]^2 + \[Beta]^2 + \[Gamma]^2)] */

                const double b =
                    -2.0 * (this->dh[0][0] * (roffset_[2] * this->dh[0][0] - y[0]) +
                            this->dh[0][1] * (roffset_[2] * this->dh[0][1] - y[1]) +
                            this->dh[0][2] * (roffset_[2] * this->dh[0][2] - y[2]));

                const double c = (roffset_[2] * this->dh[0][0] - y[0]) *
                    (roffset_[2] * this->dh[0][0] - y[0]) +
                    (roffset_[2] * this->dh[0][1] - y[1]) *
                    (roffset_[2] * this->dh[0][1] - y[1]) +
                    (roffset_[2] * this->dh[0][2] - y[2]) *
                    (roffset_[2] * this->dh[0][2] - y[2]) -
                    radius * radius;

                double delta = b * b - 4.0 * a * c;

                if (delta < 0.0)
                    continue;

                delta = sqrt(delta);

                const int xmin = std::max((int)ceil((-b - delta) * 0.5 * a_inv), lb_cube_[2]);
                const int xmax = std::min((int)floor((-b + delta) * 0.5 * a_inv), ub_cube_[2]);

                int lower_corner[3] = {iz, iy, xmin};
                int upper_corner[3] = {iz + 1, iy + 1, xmin};

                for (int x = xmin - lb_cube_[2];
                     x < std::min(xmax - lb_cube_[2], this->cube_.size(2)); x++) {
                    const int x1 = map_(2, x);

                    if (!xwindow.is_point_in_interval(x1))
                        continue;

                    compute_interval(map_.at(2, 0), this->grid_.full_size(2),
                                     this->grid_.size(2), this->cube_.size(2), x1, &x,
                                     lower_corner + 2, upper_corner + 2, xwindow);

                    if (upper_corner[2] - lower_corner[2]) {
                        const int position1[3] = {k, j, x};

                        /* the function will internally take care of the local vs global
                         * grid */
                        if (add == true) {
                            double *__restrict__ dst = this->grid_.at(lower_corner[0],
                                                                     lower_corner[1],
                                                                     lower_corner[2]);
                            const double *__restrict__ src = this->cube_.at(position1[0],
                                                                            position1[1],
                                                                            position1[2]);

                            const int sizex = upper_corner[2] - lower_corner[2];
#pragma GCC ivdep
                            for (int x1 = 0; x1 < sizex; x1++) {
                                dst[x1] += src[x1];
                            }
                        } else {
                            const double *__restrict__ const src = this->grid_.at(lower_corner[0],
                                                                                 lower_corner[1],
                                                                                 lower_corner[2]);
                            double *__restrict__ dst =
                                this->cube_.at(position1[0], position1[1], position1[2]);

                            const int sizex = upper_corner[2] - lower_corner[2];
#pragma GCC ivdep
                            for (int x1 = 0; x1 < sizex; x1++) {
                                dst[x1] = src[x1];
                            }
                        }

                        if (this->grid_.size(2) == this->grid_.full_size(2))
                            update_loop_index(this->grid_.full_size(2), x1, &x);
                        else
                            x += upper_corner[2] - lower_corner[2] - 1;
                    }
                }
            }
        }
    }

    template<bool add = true> inline void extract_add_cube() {
        set_map();

        const Interval zwindow = {.xmin = this->grid_.window_shift(0),
            .xmax = this->grid_.window_size(0)};
        const Interval ywindow = {.xmin = this->grid_.window_shift(1),
            .xmax = this->grid_.window_size(1)};
        const Interval xwindow = {.xmin = this->grid_.window_shift(2),
            .xmax = this->grid_.window_size(2)};

        int lower_corner[3];
        int upper_corner[3];
        i_z_.clear();
        i_x_.clear();
        i_y_.clear();

        for (auto z = 0; z < (int)this->cube_.size(0); z++) {
            int z1 = map_(0, z);
            if (!zwindow.is_point_in_interval(z1))
                continue;

            compute_interval(map_.at(0, 0), this->grid_.full_size(0), this->grid_.size(0),
                             this->cube_.size(0), z1, &z, lower_corner, upper_corner,
                             zwindow);

            if (upper_corner[0] - lower_corner[0]) {
                i_z_.push_back({z, lower_corner[0], upper_corner[0]});

                if (this->grid_.size(0) == this->grid_.full_size(0))
                    update_loop_index(this->grid_.full_size(0), z1, &z);
                else
                    z += upper_corner[0] - lower_corner[0] - 1;
            }
        }

        for (auto y = 0; y < (int)this->cube_.size(1); y++) {
            int y1 = map_(1, y);
            if (!ywindow.is_point_in_interval(y1))
                continue;

            compute_interval(map_.at(1, 0), this->grid_.full_size(1), this->grid_.size(1),
                             this->cube_.size(1), y1, &y, lower_corner, upper_corner,
                             ywindow);

            if (upper_corner[0] - lower_corner[0]) {
                i_y_.push_back({y, lower_corner[0], upper_corner[0]});

                if (this->grid_.size(1) == this->grid_.full_size(1))
                    update_loop_index(this->grid_.full_size(1), y1, &y);
                else
                    y += upper_corner[0] - lower_corner[0] - 1;
            }
        }

        for (auto x = 0; x < (int)this->cube_.size(2); x++) {
            int x1 = map_(2, x);
            if (!xwindow.is_point_in_interval(x1))
                continue;

            compute_interval(map_.at(2, 0), this->grid_.full_size(2), this->grid_.size(2),
                             this->cube_.size(2), x1, &x, lower_corner, upper_corner,
                             xwindow);

            if (upper_corner[0] - lower_corner[0]) {
                i_x_.push_back({x, lower_corner[0], upper_corner[0]});

                if (this->grid_.size(2) == this->grid_.full_size(2))
                    update_loop_index(this->grid_.full_size(2), x1, &x);
                else
                    x += upper_corner[0] - lower_corner[0] - 1;
            }
        }

         /* this code makes a decomposition of the cube such that we can add block of
         * datas in a vectorized way. */

        /* the decomposition depends unfortunately on the way the grid is split over
         * mpi ranks. If each mpi rank has the full grid then it is simple. An 1D
         * example of the decomposition will explain it better. We have an interval
         * [x1, x1 + cube_size - 1] (and a second index x [0, cube_size -1]) and a
         * grid that goes from [0.. grid_size - 1].
         *
         * We start from x1 and compute the largest interval [x1.. x1 + diff] that fit
         * to [0.. grid_size - 1]. Computing the difference diff is simply
         * min(grid_size - x1, cube_size - x). then we add the result in a vectorized
         * fashion. we itterate the processus by reducing the interval [x1, x1 +
         * cube_size - 1] until it is empty. */
        int position1[3];
        for (auto &z : i_z_) {
            lower_corner[0] = z.xmin;
            upper_corner[0] = z.xmax;
            position1[0] = z.x;
            for (auto &y : i_y_) {
                lower_corner[1] = y.xmin;
                upper_corner[1] = y.xmax;
                position1[1] = y.x;
                for (auto &x : i_x_) {
                    lower_corner[2] = x.xmin;
                    upper_corner[2] = x.xmax;
                    position1[2] = x.x;
                    if (add == true) {
                        // the function will internally take care of the local vx global
                        //  * grid
                        this->grid_.add_sub_grid(
                            lower_corner, // lower corner of the portion of cube (in the full grid)
                            upper_corner, // upper corner of the portion of cube (in the full grid)
                            position1, // starting position subblock inside the cube
                            this->cube_); // the grid to add data from
                    } else {
                        // the function will internally take care of the local vx global
                        //  * grid
                        this->grid_.extract_sub_grid(
                            lower_corner,
                            upper_corner,
                            position1,
                            this->cube_);
                    }
                }
            }
        }
    }

    void calculate_non_orthorombic_corrections_tensor(const double mu_mean);

    void apply_non_orthorombic_corrections(const bool *__restrict__ plane,
                                           const tensor1<double, 3> &Exp,
                                           tensor1<double, 3> &cube);

    void apply_non_orthorombic_corrections() {
        apply_non_orthorombic_corrections(orthogonal,
                                          Exp_,
                                          cube_);
    }

    void tensor_reduction(const bool integrate,
                          const double alpha,
                          tensor1<double, 3> &co,
                          tensor1<double, 3> &cube);

    void collocate_l0(const double alpha,
                      tensor1<double, 3> &cube);

    void calculate_polynomials(const bool transpose, const double dr,
                               const double roffset, const int pol_offset,
                               const int xmin, const int xmax, const int lp,
                               const int cmax, const double zetp, double *pol_);

    void apply_non_orthorombic_corrections_xy_blocked(const tensor1<double, 2> &Exp,
                                                      tensor1<double, 3> &m);

    void apply_non_orthorombic_corrections_xz_blocked(const tensor1<double, 2> &Exp,
                                                      tensor1<double, 3> &m);

    void apply_non_orthorombic_corrections_yz_blocked(const tensor1<double, 2> &Exp,
                                                      tensor1<double, 3> &m);

    void apply_non_orthorombic_corrections_xz_yz_blocked(const tensor1<double, 2> &Exp_xz,
                                                         const tensor1<double, 2> &Exp_yz,
                                                         tensor1<double, 3> &m);

    void transform_coef_jik_to_yxz(const double dh_[3][3],	tensor1<double, 3> &coef_xyz);
    void transform_coef_xzy_to_ikj(const double dh[3][3],
                                        tensor1<double, 3> &coef_xyz);
    void compute_vab(const int *const lmin,
                     const int *const lmax,
                     const int lp,
                     const double prefactor, // transformation parameters (x - x_1)^n (x -
                     // x_2)^m -> (x - x_12) ^ l
                     tensor1<double, 2> &vab);

    void transform_xyz_to_triangular(const tensor1<double, 3> &coef,
                                     double *const coef_xyz);
    void transform_yxz_to_triangular(const tensor1<double, 3> &coef,
                                     double *const coef_xyz);
    void transform_triangular_to_xyz(const double *const coef_xyz,
                                     tensor1<double, 3> &coef);
};
#endif
