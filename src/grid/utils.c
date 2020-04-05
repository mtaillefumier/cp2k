#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#if defined(__MKL) || defined(HAVE_MKL)
#include <mkl.h>
#include <mkl_cblas.h>
#endif

#ifdef __LIBXSMM
#include <libxsmm.h>
#endif

#include "utils.h"

inline int return_length_l(const int l) {
    static const int length_[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
    return length_[l];
}

inline int return_offset_l(const int l) {
    static const int offset_[10] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};
    return offset_[l];
}

inline int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma) {
    const int l = alpha + beta + gamma;
    return return_offset_l(l) + (l - alpha) * (l - alpha + 1) / 2 + gamma;
}

inline void extract_sub_grid(const int *lower_corner,
                             const int *upper_corner,
                             const int *position,
                             const tensor *grid,
                             const tensor *subgrid)
{
    for (int d = 0; d < 3; d++) {
        if ((lower_corner[d] >= grid->size[d]) ||
            (lower_corner[d] < 0) ||
            (lower_corner[d] >= upper_corner[d]) ||
            (upper_corner[d] > grid->size[d]) ||
            (upper_corner[d] <= 0) ||
            (upper_corner[d] - lower_corner[d] > subgrid->size[d]) ||
            (grid == NULL) ||
            (subgrid == NULL)) {

            printf("Error : invalid parameters. Values of the given parameters along the first wrong dimension\n");
            printf("      : lorner corner  [%d] = %d\n", d, lower_corner[d]);
            printf("      : upper  corner  [%d] = %d\n", d, upper_corner[d]);
            printf("      : diff           [%d] = %d\n", d, upper_corner[d] - lower_corner[d]);
            printf("      : src grid size  [%d] = %d\n", d, grid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, subgrid->size[d]);
            abort();
        }
    }

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
        for (int y = 0; y < sizey; y++) {
            memcpy(&idx3(subgrid[0], position1[0] + z, position1[1] + y, position1[2]),
                   &idx3(grid[0], lower_corner[0] + z, lower_corner[1] + y, lower_corner[2]),
                   sizeof(double) * sizex);
        }
    }

    return;
}

inline void add_sub_grid(const int *lower_corner,
                         const int *upper_corner,
                         const int *position,
                         const tensor *subgrid,
                         tensor *grid)
{

    int position1[3] = {0, 0, 0};

    if (position) {
        position1[0] = position[0];
        position1[1] = position[1];
        position1[2] = position[2];
    }
    for (int d = 0; d < 3; d++) {
        if ((lower_corner[d] >= grid->size[d]) ||
            (lower_corner[d] < 0) ||
            (lower_corner[d] >= upper_corner[d]) ||
            (upper_corner[d] > grid->size[d]) ||
            (upper_corner[d] <= 0) ||
            (upper_corner[d] - lower_corner[d] > subgrid->size[d]) ||
            (grid == NULL) ||
            (subgrid == NULL)) {

            printf("Error : invalid parameters. Values of the given parameters along the first wrong dimension\n");
            printf("      : lorner corner  [%d] = %d\n", d, lower_corner[d]);
            printf("      : upper  corner  [%d] = %d\n", d, upper_corner[d]);
            printf("      : diff           [%d] = %d\n", d, upper_corner[d] - lower_corner[d]);
            printf("      : src grid size  [%d] = %d\n", d, subgrid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, grid->size[d]);
            abort();
        }
    }

    const int sizex = upper_corner[2] - lower_corner[2];
    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];

    for (int z = 0; z < sizez; z++) {
        double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], lower_corner[2]);
        double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], position1[2]);
        for (int y = 0; y < sizey; y++) {
#pragma GCC unroll 4
#pragma GCC ivdep
            for (int x = 0; x < sizex; x++) {
                dst[x] += src[x];
            }
            dst += grid->ld_;
            src += subgrid->ld_;
        }
    }

    return;
}

/* add a 3D block given by subgrid and apply pcb along the fastest dimension */

void add_sub_grid_with_pcb(const int *period,
                           const int *position_inside_subgrid,
                           const int *lower_corner,
                           const int *upper_corner,
                           const tensor *subgrid,
                           tensor *grid)
{

    int position1[3] = {0, 0, 0};

    if (position_inside_subgrid) {
        position1[0] = position_inside_subgrid[0];
        position1[1] = position_inside_subgrid[1];
        position1[2] = position_inside_subgrid[2];
    }
    for (int d = 0; d < 2; d++) {
        if ((lower_corner[d] >= grid->size[d]) ||
            (lower_corner[d] < 0) ||
            (lower_corner[d] >= upper_corner[d]) ||
            (upper_corner[d] > grid->size[d]) ||
            (upper_corner[d] <= 0) ||
            (upper_corner[d] - lower_corner[d] > subgrid->size[d]) ||
            (grid == NULL) ||
            (subgrid == NULL)) {

            printf("Error : invalid parameters. Values of the given parameters along the first wrong dimension\n");
            printf("      : lorner corner  [%d] = %d\n", d, lower_corner[d]);
            printf("      : upper  corner  [%d] = %d\n", d, upper_corner[d]);
            printf("      : diff           [%d] = %d\n", d, upper_corner[d] - lower_corner[d]);
            printf("      : src grid size  [%d] = %d\n", d, subgrid->size[d]);
            printf("      : dst grid size  [%d] = %d\n", d, grid->size[d]);
            abort();
        }
    }

    const int sizey = upper_corner[1] - lower_corner[1];
    const int sizez = upper_corner[0] - lower_corner[0];

    const int offset = min(grid->size[2] - lower_corner[2], subgrid->size[2]);
    const int loop_number = (subgrid->size[2] - offset) / period[2];
    const int reminder = min(grid->size[2], subgrid->size[2] - offset - loop_number * period[2]);

    for (int z = 0; z < sizez; z++) {
        double *__restrict__ dst = &idx3(grid[0], lower_corner[0] + z, lower_corner[1], 0);
        double *__restrict__ src = &idx3(subgrid[0], position1[0] + z, position1[1], 0);
        for (int y = 0; y < sizey; y++) {
            //#pragma omp simd
            /* for (int x = 0; x < grid->size[2]; x++) */
#pragma GCC ivdep
            for (int x = 0; x < offset; x++) {
                dst[x + offset] += src[x];
            }

            int shift = offset;
            for (int l = 0; l < loop_number; l++) {
#pragma GCC ivdep
                for (int x = 0; x < grid->size[2]; x++) {
                    dst[x] += src[shift + x];
                }
                shift = offset + (l + 1) * period[2];
            }

#pragma GCC ivdep
            for (int x = 0; x < reminder; x++)
                dst[x] += src[shift + x];

            dst += grid->ld_;
            src += subgrid->ld_;
        }
    }

    return;
}


inline int compute_cube_properties(const double radius,
                                   const double dh[3][3],
                                   const double dh_inv[3][3],
                                   const double *rp,
                                   double *disr_radius,
                                   double *roffset,
                                   int *cubecenter,
                                   int *lb_cube,
                                   int *ub_cube,
                                   int *cube_size)
{
    int cmax = 0;

    /* seting up the cube parameters */
    const double dx[3] = {dh[2][2], dh[1][1], dh[0][0]};
    const double dx_inv[3] = {dh_inv[2][2], dh_inv[1][1], dh_inv[0][0]};
    /* cube center */
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        cubecenter[2 - i] = floor(dh_inv_rp);
    }

    /* lower and upper bounds */

    // Historically, the radius gets discretized.
    const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
    *disr_radius = drmin * max(1, ceil(radius/drmin));

    for (int i = 0; i < 3; i++) {
        lb_cube[i] = ceil(-1e-8 - *disr_radius * dx_inv[i]);
        ub_cube[i] = 1 - lb_cube[i];
    }

    /* compute the cube size ignoring periodicity */
    cube_size[0] = ub_cube[0] - lb_cube[0] + 1;
    cube_size[1] = ub_cube[1] - lb_cube[1] + 1;
    cube_size[2] = ub_cube[2] - lb_cube[2] + 1;

    for (int i=0; i<3; i++) {
        roffset[i] = rp[2 - i] - ((double) cubecenter[i]) * dx[i];
    }

    for (int i = 0; i < 3; i++) {
        cmax = max(cmax, ub_cube[i]);
    }

    return cmax;
}

void  return_cube_position(const int *grid_size,
                           const int *lb_grid,
                           const int *cube_center,
                           const int *lower_boundaries_cube,
                           const int *period, int *const position)
{
    position[0] = (lb_grid[0] + cube_center[0] + lower_boundaries_cube[0] + 32 * period[0]) % period[0];
    position[1] = (lb_grid[1] + cube_center[1] + lower_boundaries_cube[1] + 32 * period[1]) % period[1];
    position[2] = (lb_grid[2] + cube_center[2] + lower_boundaries_cube[2] + 32 * period[2]) % period[2];

    if ((position[0] >= grid_size[0]) || (position[1] >= grid_size[1]) || (position[2] >= grid_size[2])) {
        printf("the lower corner of the cube is outside the grid\n");
        abort();
    }
}

/* this function needs serious rethinking. Basically we apply a spherical mask
 * on a 3D grid and then add the result on a grid with PBC */

void apply_mapping(const double disr_radius,
                   const double dh[3][3],
                   const double dh_inv[3][3],
                   const int *map[3],
                   const int lb_cube[3],
                   tensor *cube,
                   const int cmax,
                   tensor *grid)
{
    const int kgmin = ceil(-1e-8 - disr_radius * dh_inv[2][2]);

    const double dz = dh[2][2];
    const double dy = dh[1][1];
    const double inv_dy = dh_inv[1][1];
    const double inv_dx = dh_inv[0][0];
    const int *__restrict map_x = map[2];
    const int *__restrict map_y = map[1];
    const int *__restrict map_z = map[0];

    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map_z[kg + cmax];   // target location on the grid
        const int kd = (2 * kg - 1) / 2;     // distance from center in grid points
        const double kr = kd * dz;   // distance from center in a.u.
        const double kremain = disr_radius * disr_radius - kr * kr;
        const int jgmin = ceil(-1e-8 - sqrt(max(0.0, kremain)) * inv_dy);
        for (int jg = jgmin; jg <= 1 - jgmin; jg++) {
            const int j = map_y[jg + cmax];  // target location on the grid
            const int jd = (2 * jg - 1) / 2;    // distance from center in grid points
            const double jr = jd * dy;  // distance from center in a.u.
            const double jremain = kremain - jr * jr;
            const int igmin = ceil(-1e-8 - sqrt(max(0.0, jremain)) * inv_dx);
            double *__restrict dst = &idx3(grid[0], k - 1, j - 1, 0);
            const double *__restrict src = &idx3(cube[0], kg - lb_cube[0], jg - lb_cube[1], - lb_cube[2]);
            for (int ig = igmin; ig <= 1 - igmin; ig++) {
                const int i = map_x[ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
            }
        }
    }
}


// *****************************************************************************
static void grid_fill_map(const bool periodic,
                          const int lb_cube,
                          const int ub_cube,
                          const int cubecenter,
                          const int lb_grid,
                          const int npts,
                          const int ngrid,
                          const int cmax,
                          int *map)
{
    if (periodic) {
        /* for (int i=0; i <= 2*cmax; i++) */
        /*     map[i] = mod(cubecenter + i - cmax, npts) + 1; */
        int start = lb_cube;
        while (true) {
            const int offset = mod(cubecenter + start, npts)  + 1 - start;
            const int length = min(ub_cube, npts - offset) - start;
            for (int ig = start; ig <= start + length; ig++) {
                map[ig + cmax] = ig + offset;
            }
            if (start + length >= ub_cube){
                break;
            }
            start += length + 1;
        }
    } else {
        // this takes partial grid + border regions into account
        const int offset = mod(cubecenter + lb_cube + lb_grid, npts) + 1 - lb_cube;
        // check for out of bounds
        assert(ub_cube + offset <= ngrid);
        assert(lb_cube + offset >= 1);
        for (int ig = lb_cube; ig <= ub_cube; ig++) {
            map[ig + cmax] = ig + offset;
        }
    }
}

/* this function needs serious rethinking. Basically we apply a spherical mask
 * on a 3D grid and then add the result on a grid with PBC */

void apply_spherical_cutoff(const double disr_radius,
                            const double dh[3][3],
                            const double dh_inv[3][3],
                            const int lb_cube[3],
                            const int cmax,
                            tensor *cube)
{
    const int kgmin = (cube->size[0] - 1) / 2;
    const int jgmin = (cube->size[1] - 1) / 2;
    const int igmin = (cube->size[2] - 1) / 2;
    const double r2 = disr_radius * disr_radius;
    const double inv_dx = 1.0 / dh[0][0];
    const double rx = dh[0][0];
    const double ry = dh[1][1];
    const double rz = dh[2][2];

    for (int kg = -kgmin; kg <= kgmin + 1; kg++) {
        double kr = ((double)kg) * rz;   // distance from center in a.u.
        kr *= kr;
        for (int jg = -jgmin; jg <= jgmin + 1; jg++) {
            double jr = ((double)jg) * ry;   // distance from center in a.u.
            jr *= jr;
            const double rest = r2 - kr - jr;
            if (rest < 1e-8) {
                memset(&idx3(cube[0], kg + kgmin, jg + jgmin, 0), 0, cube->size[2] * sizeof(double));
                continue;
            }
            double *__restrict dst =  &idx3(cube[0], kg + kgmin, jg + jgmin, 0);
            int rix = ceil(-1e-8 - sqrt(max(0.0, rest)) * inv_dx);
            for (int ig = -igmin; ig <= rix; ig ++) {
                dst[ig + igmin] = 0.0;
            }
            for (int ig = -rix + 1; ig < cube->size[2]; ig ++) {
                dst[ig + igmin] = 0.0;
            }
        }
    }
}


void compute_folded_polynomial(const int cube_center,
                               const int pol_length,
                               const int period,
                               const int grid_size,
                               const int grid_lower_boundaries,
                               const int *__restrict non_zero_elements,
                               const double *__restrict pol,
                               double *__restrict res,
                               double *__restrict tmp)
{

    const int start = (cube_center - grid_lower_boundaries - (pol_length - 1) / 2 + 32 * period) % period;

#pragma GCC unroll 4
#pragma GCC ivdep
    for (int s = start; s < min(grid_size, pol_length + start); s++) {
        tmp[s] = pol[s - start];
    }

    for (int s = min(grid_size - start, pol_length); s < pol_length; s++) {
#pragma GCC unroll 4
#pragma GCC ivdep
        for (int s1 = 0; s1 < min(grid_size, pol_length - s); s1++)
            tmp[s1] += pol[s + s1];

        s += period;
    }

    // now compress the all thing. I do not care about zeros.
    int offset = 0;

    for (int s = 0; s < grid_size; s++) {
        /* if (non_zero_elements[s] == 0) */
        /*     continue; */
        for (; (s < grid_size - 1) && (non_zero_elements[s] == 0); s++);
        int smin = s;

        for (; (s < grid_size - 1) && (non_zero_elements[s] == 1); s++);

        int smax = s + non_zero_elements[s];

        for (int si = 0; si < (smax - smin); si++)
            res[offset + si] = tmp[smin + si];
        offset += smax - smin;
    }
}


void calculate_non_orthorombic_corrections_tensor(const double*__restrict r_ab,
                                                  const double mu_mean,
                                                  const double *dr,
                                                  const double basis[3][3],
                                                  const int *corner,
                                                  const int *period,
                                                  tensor *const Exp)
{
    // need to move this outside
    /* void *tmp = Exp->data; */
    /* realloc_tensor(tmp, */
    /*                handler->alpha_alloc_size, */
    /*                handler->alpha.alloc_size_, */
    /*                (void **)&(Exp->data)); */

    const int n[3][2] = {{0, 2},
                         {0, 1},
                         {1, 2}};

    // need to review this
    const double c[3] = {
        /* beta gamma */
        -2.0 * (basis[1][0] * basis[2][0] + basis[1][1] * basis[2][1] + basis[1][2] * basis[2][2]),
        /* alpha gamma */
        -2.0 * (basis[0][0] * basis[2][0] + basis[0][1] * basis[2][1] + basis[0][2] * basis[2][2]),
        /* alpha beta */
        -2.0 * (basis[0][0] * basis[1][0] + basis[0][1] * basis[1][1] + basis[0][2] * basis[1][2])
    };

    for (int dir = 0; dir < 3; dir++) {
        int d1 = n[dir][0];
        int d2 = n[dir][1];

        const double coef = c[dir];

        for (int alpha = 0; alpha < Exp->size[d1]; alpha++) {
            double alpha_d = (alpha + corner[d1]) * dr[d1] - period[d1] * dr[d1] / 2.0 - r_ab[d1];
            for (int beta = 0; beta < Exp->size[d2]; beta++) {
                int beta_d = (beta + corner[d2]) * dr[d2] - period[d2] * dr[d2] / 2.0  - r_ab[d2];
                idx3(Exp[0], dir, alpha, beta) = exp(-coef * mu_mean * alpha_d * beta_d);
            }
        }
    }
}
