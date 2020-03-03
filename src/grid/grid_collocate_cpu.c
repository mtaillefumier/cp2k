/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <assert.h>
#include <string.h>
#include <mkl_cblas.h>
#include <libxsmm.h>
#include "grid_collocate_replay.h"
#include "grid_collocate_cpu.h"
#include "grid_prepare_pab.h"
#include "grid_common.h"
#include "tensor_local.h"

// *****************************************************************************
static void grid_prepare_alpha(const double ra[3],
                               const double rb[3],
                               const double rp[3],
                               const int la_max,
                               const int lb_max,
                               tensor *alpha) // [3][lb_max+1][la_max+1][la_max+lb_max+1]
{
    // Initialize with zeros.
    memset(alpha->data, 0, alpha->alloc_size_ * sizeof(double));

    //
    //   compute polynomial expansion coefs -> (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
    //

    for (int iaxis=0; iaxis<3; iaxis++) {
        const double drpa = rp[iaxis] - ra[iaxis];
        const double drpb = rp[iaxis] - rb[iaxis];
        for (int lxa=0; lxa<=la_max; lxa++) {
            for (int lxb=0; lxb<=lb_max; lxb++) {
                double binomial_k_lxa = 1.0;
                double a = 1.0;
                for (int k=0; k<=lxa; k++) {
                    double binomial_l_lxb = 1.0;
                    double b = 1.0;
                    for (int l=0; l<=lxb; l++) {
                        idx4(alpha[0], iaxis, lxb, lxa, lxa-l+lxb-k) += binomial_k_lxa * binomial_l_lxb * a * b;
                        binomial_l_lxb *= ((double)(lxb - l)) / ((double)(l + 1));
                        b *= drpb;
                    }
                    binomial_k_lxa *= ((double)(lxa-k)) / ((double)(k+1));
                    a *= drpa;
                }
            }
        }
    }
}

// *****************************************************************************
static void grid_prepare_coef(const int la_max,
                              const int la_min,
                              const int lb_max,
                              const int lb_min,
                              const int lp,
                              const double prefactor,
                              const tensor *alpha, // [3][lb_max+1][la_max+1][lp+1]
                              const double pab[ncoset[lb_max]][ncoset[la_max]],
                              tensor *coef_xyz) //[lp+1][lp+1][lp+1]
{

    memset(coef_xyz->data, 0, coef_xyz->alloc_size_ * sizeof(double));

    double coef_xyt[lp+1][lp+1];
    double coef_xtt[lp+1];

    for (int lzb = 0; lzb<=lb_max; lzb++) {
    for (int lza = 0; lza<=la_max; lza++) {
       for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
          for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
             coef_xyt[lyp][lxp] = 0.0;
          }
       }
       for (int lyb = 0; lyb<=lb_max-lzb; lyb++) {
       for (int lya = 0; lya<=la_max-lza; lya++) {
          const int lxpm = (lb_max-lzb-lyb) + (la_max-lza-lya);
          for (int i=0; i<=lxpm; i++) {
              coef_xtt[i] = 0.0;
          }
          for (int lxb = max(lb_min-lzb-lyb, 0); lxb<=lb_max-lzb-lyb; lxb++) {
          for (int lxa = max(la_min-lza-lya, 0); lxa<=la_max-lza-lya; lxa++) {
             const int ico = coset(lxa, lya, lza);
             const int jco = coset(lxb, lyb, lzb);
             const double p_ele = prefactor * pab[jco][ico];
             for (int lxp = 0; lxp<=lxa+lxb; lxp++) {
                 coef_xtt[lxp] += p_ele * idx4(alpha[0], 0, lxb, lxa, lxp);
             }
          }
          }
          for (int lyp = 0; lyp<=lya+lyb; lyp++) {
             for (int lxp = 0; lxp<=lp-lza-lzb-lya-lyb; lxp++) {
                 coef_xyt[lyp][lxp] += idx4(alpha[0], 1, lyb, lya, lyp) * coef_xtt[lxp];
             }
          }
       }
       }
       for (int lzp = 0; lzp<=lza+lzb; lzp++) {
          for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
             for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                 idx3(coef_xyz[0], lzp, lyp, lxp) += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
             }
          }
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
                          int map[2*cmax + 1])
{
    if (periodic) {
        //for (int i=0; i <= 2*cmax; i++)
        //    map[i] = mod(cubecenter + i - cmax, npts) + 1;
        int start = lb_cube;
        while (true) {
            const int offset = mod(cubecenter + start, npts)  + 1 - start;
            const int length = min(ub_cube, npts - offset) - start;
            for (int ig=start; ig<=start+length; ig++) {
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
        for (int ig=lb_cube; ig <= ub_cube; ig++) {
            map[ig + cmax] = ig + offset;
        }
    }
}


// *****************************************************************************
static void grid_fill_pol(const double dr,
                          const double roffset,
                          const int lb_cube,
                          const int lp,
                          const int cmax,
                          const double zetp,
                          double *pol_)
{
    tensor pol;
    initialize_tensor_2(&pol, lp + 1, 2 * cmax + 1);
    pol.data = pol_;
//
//   compute the values of all (x-xp)**lp*exp(..)
//
//  still requires the old trick:
//  new trick to avoid to many exps (reuse the result from the previous gridpoint):
//  exp( -a*(x+d)**2)=exp(-a*x**2)*exp(-2*a*x*d)*exp(-a*d**2)
//  exp(-2*a*(x+d)*d)=exp(-2*a*x*d)*exp(-2*a*d**2)
//
      const double t_exp_1 = exp(-zetp * pow(dr, 2));
      const double t_exp_2 = pow(t_exp_1, 2);

      double t_exp_min_1 = exp(-zetp * pow(+dr - roffset, 2));
      double t_exp_min_2 = exp(-2 * zetp * (+dr - roffset) * (-dr));
      for (int ig=0; ig >= lb_cube; ig--) {
          const double rpg = ig * dr - roffset;
          t_exp_min_1 *= t_exp_min_2 * t_exp_1;
          t_exp_min_2 *= t_exp_2;
          double pg = t_exp_min_1;
          // pg  = EXP(-zetp*rpg**2)
          for (int icoef=0; icoef <= lp; icoef++) {
              idx2(pol, icoef, ig - lb_cube) = pg;
              pg *= rpg;
          }
      }

      double t_exp_plus_1 = exp(-zetp * pow(-roffset,2));
      double t_exp_plus_2 = exp(-2 * zetp * (-roffset) * (+dr));
      for (int ig=0; ig >= lb_cube; ig--) {
          const double rpg = (1-ig) * dr - roffset;
          t_exp_plus_1 *= t_exp_plus_2 * t_exp_1;
          t_exp_plus_2 *= t_exp_2;
          double pg = t_exp_plus_1;
          // pg  = EXP(-zetp*rpg**2)
          for (int icoef=0; icoef<=lp; icoef++) {
              idx2(pol, icoef, 1 - ig - lb_cube) = pg;
              pg *= rpg;
          }
      }
}

void collocate_core_rectangular(char *scratch,
                                const int length_[3],
                                const struct tensor_ *co,
                                const struct tensor_ *p_alpha_beta_reduced_,
                                struct tensor_ *Vtmp)
{
    if (co->size[0] > 1) {
        tensor C;
        tensor xyz_alpha_beta;

        initialize_tensor_3(&C, co->size[0], co->size[1], length_[1]);

        initialize_tensor_3(&xyz_alpha_beta, co->size[1], length_[0], length_[1]);

#if defined(LIBXSMM)
        C.data = libxsmm_aligned_scratch(sizeof(double) * C.alloc_size_, 0/*auto-alignment*/);
        xyz_alpha_beta.data = libxsmm_aligned_scratch(sizeof(double) * xyz_alpha_beta.alloc_size_, 0/*auto-alignment*/);
#else
        C.data = (double *)scratch;
        xyz_alpha_beta.data = ((double *)scratch) + C.alloc_size_ * sizeof(double);
#endif


// we can batch this easily
        // for (int a1 = 0; a1 < co.size(0); a1++) {
        // we need to replace this with libxsmm
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    co->size[0] * co->size[1],
                    length_[1],
                    co->size[2],
                    1.0,
                    co->data, // Coef_{alpha,gamma,beta}
                    co->ld_,
                    p_alpha_beta_reduced_->data + p_alpha_beta_reduced_->offsets[0], // Y_{beta,j} p_alpha_beta_reduced(1, 0, 0)
                    p_alpha_beta_reduced_->ld_,
                    0.0,
                    C.data, // tmp_{alpha, gamma, j}
                    C.ld_);
        // }

        for (int a1 = 0; a1 < co->size[0]; a1++) {
            cblas_dgemm(CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        length_[0],
                        length_[1],
                        co->size[2],
                        1.0,
                        p_alpha_beta_reduced_->data, // I start from (0,0,0) Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
                        p_alpha_beta_reduced_->ld_,
                        C.data + a1 * C.offsets[0], // &idx3(C, a1, 0, 0), // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
                        C.ld_,
                        0.0,
                        xyz_alpha_beta.data + a1 * xyz_alpha_beta.offsets[0], // contains xyz_{alpha, kj} the order kj is important
                        xyz_alpha_beta.ld_);
        }

        cblas_dgemm(CblasRowMajor,
                    CblasTrans,
                    CblasNoTrans,
                    length_[0] * length_[1],
                    length_[2],
                    co->size[2],
                    1.0,
                    &idx3(xyz_alpha_beta, 0, 0, 0),
                    xyz_alpha_beta.size[1] * xyz_alpha_beta.ld_,
                    &idx3(p_alpha_beta_reduced_[0], 2, 0, 0),
                    p_alpha_beta_reduced_->ld_,
                    0.0,
                    &idx3(Vtmp[0], 0, 0, 0),
                    Vtmp->ld_);

#if defined(SCRATCH)
        libxsmm_free(C.data);
        libxsmm_free(xyz_alpha_beta.data);
#endif

    } else {
        const double *__restrict pz = &idx3(p_alpha_beta_reduced_[0], 2, 0, 0);
        const double *__restrict py = &idx3(p_alpha_beta_reduced_[0], 1, 0, 0);
        const double *__restrict px = &idx3(p_alpha_beta_reduced_[0], 0, 0, 0);
        double *__restrict dst = &idx3(Vtmp[0], 0, 0, 0);
        const double coo = idx3 (co[0], 0, 0, 0);
        for (int z1 = 0; z1 < length_[0]; z1++) {
            const double tz = coo * pz[z1];
            for (int y1 = 0; y1 < length_[1]; y1++) {
                const double tmp = tz * py[y1];
                for (int x1 = 0; x1 < length_[2]; x1++) {
                    dst[x1] = tmp * px[x1];
                }
                dst += Vtmp->ld_;
            }
        }
    }
}

#if defined(HANS_ALGO)
void collocate_core_hans(char *scratch,
                         const int length_[3],
                         const tensor *co,
                         const tensor *p_alpha_beta_reduced_,
                         tensor *Vtmp)
{

    tensor C, xyz_alpha_beta;
    initialize_tensor_2(&C, co.size(1), length_[1]);
    initialize_tensor_3(&xyz_alpha_beta, co.size(0), length_[0], length_[1]);

#if defined(libxsmm)
    C.data = libxsmm_aligned_scratch(sizeof(T) * co.size(0) * co.size(1) * length_[1], 0/*auto-alignment*/);
    xyz_alpha_beta.data = libxsmm_aligned_scratch(sizeof(T) * co.size(0) * length_[0] * length_[1], 0/*auto-alignment*/);
#else
    C.data = (double *)
        mdarray<T, 3, CblasRowMajor> C(co.size(0), co.size(1), length_[1]);
    mdarray<T, 3, CblasRowMajor> xyz_alpha_beta(co.size(0), length_[0], length_[1]);
#endif
    const T *LIBXSMM_RESTRICT abr0 = p_alpha_beta_reduced_.template at<CPU>(0, 0, 0);
    const T *LIBXSMM_RESTRICT abr1 = p_alpha_beta_reduced_.template at<CPU>(1, 0, 0);
    const T *LIBXSMM_RESTRICT abr2 = p_alpha_beta_reduced_.template at<CPU>(2, 0, 0);
#if defined(XSMM)
  const libxsmm_mmfunction<T> xmm1(LIBXSMM_GEMM_FLAG_NONE, length_[1], co.size(2), co.size(2),
    p_alpha_beta_reduced_.ld(), co.ld(), /*C.ld()*/length_[1],
    1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_AUTO);
  const libxsmm_mmfunction<T> xmm2(LIBXSMM_GEMM_FLAG_TRANS_B, length_[1], length_[0], co.size(2),
    /*C.ld()*/length_[1], p_alpha_beta_reduced_.ld(), /*xyz_alpha_beta.ld()*/length_[1],
    1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_AUTO);
  const libxsmm_mmfunction<T> xmm3(LIBXSMM_GEMM_FLAG_TRANS_B, length_[2], length_[0] * length_[1], co.size(2),
    p_alpha_beta_reduced_.ld(), /*xyz_alpha_beta.size(1)*/length_[0] * /*xyz_alpha_beta.ld()*/length_[1], Vtmp.ld(),
    1/*alpha*/, 0/*beta*/, LIBXSMM_PREFETCH_NONE);
#endif
  timer.stop("init");

  if (co.size(0) > 1) {
    timer.start("gemm");
    const T* bj = co.template at<CPU>(0, 0, 0);
#if defined(SCRATCH)
    T* cj = &LIBXSMM_VLA_ACCESS(3, C, 0, 0, 0, co.size(1), length_[1]);
#else
    T* cj = C.template at<CPU>(0, 0, 0);
#endif
    // run loop excluding the last element
    for (int a1 = 0; a1 < static_cast<int>(co.size(0) - 1); a1++) {
      const T *const bi = bj; bj = co.template at<CPU>(a1 + 1, 0, 0);
#if defined(SCRATCH)
      T *const ci = cj; cj = &LIBXSMM_VLA_ACCESS(3, C, a1 + 1, 0, 0, co.size(1), length_[1]);
#else
      T *const ci = cj; cj = C.template at<CPU>(a1 + 1, 0, 0);
#endif
#if defined(XSMM)
      xmm1(abr1, bi, ci, abr1, bj, cj);
#else
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        co.size(2), length_[1], co.size(2),
        1.0,
        bi, // Coef_{alpha,gamma,beta}
        co.ld(),
        abr1, // Y_{beta,j}
        p_alpha_beta_reduced_.ld(),
        0.0,
        ci, // tmp_{alpha, gamma, j}
        /*C.ld()*/length_[1]);
#endif
    }
    // execute remainder with pseudo-prefetch
#if defined(XSMM)
    xmm1(abr1, bj, cj, abr1, bj, cj);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      co.size(2), length_[1], co.size(2),
      1.0,
      bj, // Coef_{alpha,gamma,beta}
      co.ld(),
      abr1, // Y_{beta,j}
      p_alpha_beta_reduced_.ld(),
      0.0,
      cj, // tmp_{alpha, gamma, j}
      /*C.ld()*/length_[1]);
#endif

    // run loop excluding the last element
#if defined(SCRATCH)
    T* aj = &LIBXSMM_VLA_ACCESS(3, C, 0, 0, 0, co.size(1), length_[1]);
    cj = &LIBXSMM_VLA_ACCESS(3, xyz_alpha_beta, 0, 0, 0, length_[0], length_[1]);
#else
    T* aj = C.template at<CPU>(0, 0, 0);
    cj = xyz_alpha_beta.template at<CPU>(0, 0, 0);
#endif
    for (int a1 = 0; a1 < static_cast<int>(co.size(0) - 1); a1++) {
      T *const ai = aj, *const ci = cj;
#if defined(SCRATCH)
      aj = &LIBXSMM_VLA_ACCESS(3, C, a1 + 1, 0, 0, co.size(1), length_[1]);
      cj = &LIBXSMM_VLA_ACCESS(3, xyz_alpha_beta, a1 + 1, 0, 0, length_[0], length_[1]);
#else
      aj = C.template at<CPU>(a1 + 1, 0, 0);
      cj = xyz_alpha_beta.template at<CPU>(a1 + 1, 0, 0);
#endif
#if defined(XSMM)
      xmm2(ai, abr0, ci, aj, abr0, cj);
#else
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        length_[0], length_[1], co.size(2),
        1.0,
        abr0, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
        p_alpha_beta_reduced_.ld(),
        ai, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
        /*C.ld()*/length_[1],
        0.0,
        ci, // contains xyz_{alpha, kj} the order kj is important
        /*xyz_alpha_beta.ld()*/length_[1]);
#endif
    }
    // execute remainder with pseudo-prefetch
#if defined(XSMM)
    xmm2(aj, abr0, cj, aj, abr0, cj);
#else
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      length_[0], length_[1], co.size(2),
      1.0,
      abr0, // Z_{gamma,k} -> I need to transpose it I want Z_{k,gamma}
      p_alpha_beta_reduced_.ld(),
      aj, // C_{gamma, j} = Coef_{alpha,gamma,beta} Y_{beta,j} (fixed alpha)
      /*C.ld()*/length_[1],
      0.0,
      cj, // contains xyz_{alpha, kj} the order kj is important
      /*xyz_alpha_beta.ld()*/length_[1]);
#endif

#if defined(SCRATCH)
    cj = &LIBXSMM_VLA_ACCESS(3, xyz_alpha_beta, 0, 0, 0, length_[0], length_[1]);
#else
    cj = xyz_alpha_beta.template at<CPU>(0, 0, 0);
#endif
#if defined(XSMM)
    xmm3(abr2, cj, Vtmp.template at<CPU>(0, 0, 0));
#else
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      length_[0] * length_[1], length_[2], co.size(2),
      1.0,
      cj,
      /*xyz_alpha_beta.size(1)*/length_[0] * /*xyz_alpha_beta.ld()*/length_[1],
      abr2,
      p_alpha_beta_reduced_.ld(),
      0.0,
      Vtmp.template at<CPU>(0, 0, 0),
      Vtmp.ld());
#endif
    timer.stop("gemm");
#if defined(SCRATCH)
    timer.start("deinit");
    libxsmm_free(Cdata);
    libxsmm_free(xyz_data);
    timer.stop("deinit");
#endif
  } else {
    for (int z1 = 0; z1 < length_[0]; z1++) {
      const T tz = co(0, 0, 0) * p_alpha_beta_reduced_(0, 0, z1);
      for (int y1 = 0; y1 < length_[1]; y1++) {
        const T tmp = tz * p_alpha_beta_reduced_(1, 0, y1);
        const T *LIBXSMM_RESTRICT src = abr2;
        T *LIBXSMM_RESTRICT dst = Vtmp.template at<CPU>(z1, y1, 0);
        for (int x1 = 0; x1 < length_[2]; x1++) {
          dst[x1] = tmp * src[x1];
        }
      }
    }
  }
}
#endif

// *****************************************************************************
static void grid_collocate_core(const int lp,
                                const int cmax,
                                const tensor *coef_xyz, // [lp+1][lp+1][lp+1]
                                const tensor *pol,
                                const int map[3][2*cmax+1],
                                const int lb_cube[3],
                                const int ub_cube[3],
                                const double dh[3][3],
                                const double dh_inv[3][3],
                                const double disr_radius,
                                const int ngrid[3],
                                double grid[ngrid[2]][ngrid[1]][ngrid[0]]) {

    // Create the full cube, ignoring periodicity for now.
    const int nz = ub_cube[2] - lb_cube[2] + 1;
    const int ny = ub_cube[1] - lb_cube[1] + 1;
    const int nx = ub_cube[0] - lb_cube[0] + 1;

    int size_[3] = {nz, ny, nx};
    tensor cube;
    initialize_tensor_3(&cube, nz, ny, nx);

#if defined(LIBXSMM)
    cube.data =  libxsmm_aligned_scratch(sizeof(double) * cube.alloc_size_, 0/*auto-alignment*/);
    char *tmp = NULL;
#else
    cube.data = (double*)scratch;
    char *tmp = scratch + cube.alloc_size_ * sizeof(double);
#endif
    memset(cube.data, 0, cube.alloc_size_ * sizeof(double));

    /* now we can work on collocate separately from the rest using tensor
     * structure. We need to include Hans rework (C++ -> C)
     */

    collocate_core_rectangular(NULL, // will need to change that eventually
                               size_,
                               coef_xyz,
                               pol,
                               &cube);

    //
    // Write cube back to large grid taking periodicity and radius into account.
    //

    // The cube contains an even number of grid points in each direction and
    // collocation is always performed on a pair of two opposing grid points.
    // Hence, the points with index 0 and 1 are both assigned distance zero via
    // the formular distance=(2*index-1)/2.

    const int kgmin = ceil(-1e-8 - disr_radius * dh_inv[2][2]);
    for (int kg = kgmin; kg <= 1 - kgmin; kg++) {
        const int k = map[2][kg + cmax];   // target location on the grid
        const int kd = (2 * kg - 1) / 2;     // distance from center in grid points
        const double kr = kd * dh[2][2];   // distance from center in a.u.
        const double kremain = disr_radius * disr_radius - kr * kr;
        const int jgmin = ceil(-1e-8 - sqrt(max(0.0, kremain)) * dh_inv[1][1]);
        for (int jg = jgmin; jg <= 1 - jgmin; jg++) {
            const int j = map[1][jg + cmax];  // target location on the grid
            const int jd = (2*jg - 1) / 2;    // distance from center in grid points
            const double jr = jd * dh[1][1];  // distance from center in a.u.
            const double jremain = kremain - jr * jr;
            const int igmin = ceil(-1e-8 - sqrt(max(0.0, jremain)) * dh_inv[0][0]);
            double *__restrict dst = &grid[k-1][j-1][0];
            const double *__restrict src = &idx3(cube, kg - lb_cube[2], jg - lb_cube[1], - lb_cube[0]);
            for (int ig=igmin; ig<=1-igmin; ig++) {
                const int i = map[0][ig + cmax];  // target location on the grid
                dst[i - 1] += src[ig];
            }
        }
    }
}

// *****************************************************************************
static void grid_collocate_ortho(const int lp,
                                 const double zetp,
                                 const tensor *coef_xyz, // [lp+1][lp+1][lp+1]
                                 const double dh[3][3],
                                 const double dh_inv[3][3],
                                 const double rp[3],
                                 const int npts[3],
                                 const int lb_grid[3],
                                 const bool periodic[3],
                                 const double radius,
                                 const int ngrid[3],
                                 double grid[ngrid[2]][ngrid[1]][ngrid[0]]) {

   // *** position of the gaussian product
   //
   // this is the actual definition of the position on the grid
   // i.e. a point rp(:) gets here grid coordinates
   // MODULO(rp(:)/dr(:),npts(:))+1
   // hence (0.0,0.0,0.0) in real space is rsgrid%lb on the rsgrid ((1,1,1) on grid)

    // cubecenter(:) = FLOOR(MATMUL(dh_inv, rp))
    int cubecenter[3];
    for (int i=0; i<3; i++) {
        double dh_inv_rp = 0.0;
        for (int j=0; j<3; j++) {
            dh_inv_rp += dh_inv[j][i] * rp[j];
        }
        cubecenter[i] = floor(dh_inv_rp);
    }

    double roffset[3];
    for (int i=0; i<3; i++) {
        roffset[i] = rp[i] - ((double) cubecenter[i]) * dh[i][i];
    }

    // Historically, the radius gets discretized.
    const double drmin = min(dh[0][0], min(dh[1][1], dh[2][2]));
    const double disr_radius = drmin * max(1, ceil(radius/drmin));

    int lb_cube[3], ub_cube[3];
    for (int i=0; i<3; i++) {
        lb_cube[i] = ceil(-1e-8 - disr_radius * dh_inv[i][i]);
        ub_cube[i] = 1 - lb_cube[i];
    }

    //cmax = MAXVAL(ub_cube)
    int cmax = INT_MIN;
    for (int i=0; i<3; i++) {
        cmax = max(cmax, ub_cube[i]);
    }

    // a mapping so that the ig corresponds to the right grid point
    int map[3][2*cmax+1];
    for (int i=0; i<3; i++) {
        grid_fill_map(periodic[i],
                      lb_cube[i],
                      ub_cube[i],
                      cubecenter[i],
                      lb_grid[i],
                      npts[i],
                      ngrid[i],
                      cmax,
                      map[i]);
    }

    /* double pol[3][lp+1][2*cmax+1]; */
    tensor pol;
    initialize_tensor_3(&pol, 3, lp + 1, 2 * cmax + 1);

#if defined(LIBXSMM)
    pol.data =  libxsmm_aligned_scratch(sizeof(double) * pol.alloc_size_, 0/*auto-alignment*/);
#else
    pol.data = (double*)tmp;
    tmp += poly.alloc_size_ * sizeof(double);
#endif

    for (int i=0; i<3; i++) {
        grid_fill_pol(dh[i][i], roffset[i], lb_cube[i], lp, cmax, zetp, pol.data + i * pol.size[1] * pol.ld_);
    }

    grid_collocate_core(lp,
                        cmax,
                        coef_xyz,
                        &pol,
                        map,
                        lb_cube,
                        ub_cube,
                        dh,
                        dh_inv,
                        disr_radius,
                        ngrid,
                        grid);

#if defined(LIBXSMM)
    libxsmm_free(pol.data);
#endif
}


// *****************************************************************************
static void grid_collocate_general(const int lp,
                                   const double zetp,
                                   const tensor *coef_xyz, // [lp+1][lp+1][lp+1]
                                   const double dh[3][3],
                                   const double dh_inv[3][3],
                                   const double rp[3],
                                   const int npts[3],
                                   const int lb_grid[3],
                                   const bool periodic[3],
                                   const double radius,
                                   const int ngrid[3],
                                   double grid[ngrid[2]][ngrid[1]][ngrid[0]]) {

// Translated from collocate_general_opt()
//
// transform P_{lxp,lyp,lzp} into a P_{lip,ljp,lkp} such that
// sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-x_p)**lxp (y-y_p)**lyp (z-z_p)**lzp =
// sum_{lip,ljp,lkp} P_{lip,ljp,lkp} (i-i_p)**lip (j-j_p)**ljp (k-k_p)**lkp
//

    // aux mapping array to simplify life
    //TODO instead of this map we could use 3D arrays like coef_xyz.
    int coef_map[lp+1][lp+1][lp+1];

    //TODO really needed?
    //coef_map = HUGE(coef_map)
    for (int lzp=0; lzp<=lp; lzp++) {
        for (int lyp=0; lyp<=lp; lyp++) {
            for (int lxp=0; lxp<=lp; lxp++) {
                coef_map[lzp][lyp][lxp] = INT_MAX;
            }
        }
    }

    int lxyz = 0;
    for (int lzp=0; lzp<=lp; lzp++) {
        for (int lyp=0; lyp<=lp-lzp; lyp++) {
            for (int lxp=0; lxp<=lp-lzp-lyp; lxp++) {
                coef_map[lzp][lyp][lxp] = ++lxyz;
            }
        }
    }

    // center in grid coords
    // gp = MATMUL(dh_inv, rp)
    double gp[3];
    for (int i=0; i<3; i++) {
        gp[i] = 0.0;
        for (int j=0; j<3; j++) {
            gp[i] += dh_inv[j][i] * rp[j];
        }
    }

    // transform using multinomials
    double hmatgridp[lp+1][3][3];
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            hmatgridp[0][j][i] = 1.0;
            for (int k=1; k<=lp; k++) {
                hmatgridp[k][j][i] = hmatgridp[k-1][j][i] * dh[j][i];
            }
        }
    }

    // zero coef_ijk
    const int ncoef_ijk = ((lp+1)*(lp+2)*(lp+3))/6;
    double coef_ijk[ncoef_ijk];
    for (int i=0; i<ncoef_ijk; i++) {
        coef_ijk[i] = 0.0;
    }

    const int lpx = lp;
    for (int klx=0; klx<=lpx; klx++) {
    for (int jlx=0; jlx<=lpx-klx; jlx++) {
    for (int ilx=0; ilx<=lpx-klx-jlx; ilx++) {
        const int lx = ilx + jlx + klx;
        const int lpy = lp - lx;
        for (int kly=0; kly<=lpy; kly++) {
        for (int jly=0; jly<=lpy-kly; jly++) {
        for (int ily=0; ily<=lpy-kly-jly; ily++) {
            const int ly = ily + jly + kly;
            const int lpz = lp - lx - ly;
            for (int klz=0; klz<=lpz; klz++) {
            for (int jlz=0; jlz<=lpz-klz; jlz++) {
            for (int ilz=0; ilz<=lpz-klz-jlz; ilz++) {
                const int lz = ilz + jlz + klz;
                const int il = ilx + ily + ilz;
                const int jl = jlx + jly + jlz;
                const int kl = klx + kly + klz;
                const int lijk= coef_map[kl][jl][il];
                coef_ijk[lijk-1] += idx3(coef_xyz[0], lz, ly, lx) *
                   hmatgridp[ilx][0][0] * hmatgridp[jlx][1][0] * hmatgridp[klx][2][0] *
                   hmatgridp[ily][0][1] * hmatgridp[jly][1][1] * hmatgridp[kly][2][1] *
                   hmatgridp[ilz][0][2] * hmatgridp[jlz][1][2] * hmatgridp[klz][2][2] *
                   fac[lx] * fac[ly] * fac[lz] /
                   (fac[ilx] * fac[ily] * fac[ilz] * fac[jlx] * fac[jly] * fac[jlz] * fac[klx] * fac[kly] * fac[klz]);
            }
            }
            }
        }
        }
        }
    }
    }
    }

    // CALL return_cube_nonortho(cube_info, radius, index_min, index_max, rp)
    //
    // get the min max indices that contain at least the cube that contains a sphere around rp of radius radius
    // if the cell is very non-orthogonal this implies that many useless points are included
    // this estimate can be improved (i.e. not box but sphere should be used)
    int index_min[3], index_max[3];
    for (int idir=0; idir<3; idir++) {
        index_min[idir] = INT_MAX;
        index_max[idir] = INT_MIN;
    }
    for (int i=-1; i<=1; i++) {
    for (int j=-1; j<=1; j++) {
    for (int k=-1; k<=1; k++) {
       const double x = rp[0] + i * radius;
       const double y = rp[1] + j * radius;
       const double z = rp[2] + k * radius;
       for (int idir=0; idir<3; idir++) {
          const double resc = dh_inv[0][idir] * x + dh_inv[1][idir] * y + dh_inv[2][idir] * z;
          index_min[idir] = min(index_min[idir], floor(resc));
          index_max[idir] = max(index_max[idir], ceil(resc));
       }
    }
    }
    }

    int offset[3];
    for (int idir=0; idir<3; idir++) {
        offset[idir] = mod(index_min[idir] + lb_grid[idir], npts[idir]) + 1;
    }

    // go over the grid, but cycle if the point is not within the radius
    for (int k=index_min[2]; k<=index_max[2]; k++) {
       const double dk = k - gp[2];
       int k_index;
       if (periodic[2]) {
          k_index = mod(k, npts[2]) + 1;
       } else {
          k_index = k - index_min[2] + offset[2];
       }

       // zero coef_xyt
       const int ncoef_xyt = ((lp+1)*(lp+2))/2;
       double coef_xyt[ncoef_xyt];
       for (int i=0; i<ncoef_xyt; i++) {
           coef_xyt[i] = 0.0;
       }

       int lxyz = 0;
       double dkp = 1.0;
       for (int kl=0; kl<=lp; kl++) {
          int lxy = 0;
          for (int jl=0; jl<=lp-kl; jl++) {
             for (int il=0; il<=lp-kl-jl; il++) {
                coef_xyt[lxy++] += coef_ijk[lxyz++] * dkp;
             }
             lxy += kl;
          }
          dkp *= dk;
       }


       for (int j=index_min[1]; j<=index_max[1]; j++) {
          const double dj = j - gp[1];
          int j_index;
          if (periodic[1]) {
             j_index = mod(j, npts[1]) + 1;
          } else {
             j_index = j - index_min[1] + offset[1];
          }

          double coef_xtt[lp+1];
          for (int i=0; i<=lp; i++) {
              coef_xtt[i] = 0.0;
          }
          int lxy = 0;
          double djp = 1.0;
          for (int jl=0; jl<=lp; jl++) {
             for (int il=0; il<=lp-jl; il++) {
                coef_xtt[il] += coef_xyt[lxy++] * djp;
             }
             djp *= dj;
          }

          // find bounds for the inner loop
          // based on a quadratic equation in i
          // a*i**2+b*i+c=radius**2

          // v = pointj-gp(1)*hmatgrid(:, 1)
          // a = DOT_PRODUCT(hmatgrid(:, 1), hmatgrid(:, 1))
          // b = 2*DOT_PRODUCT(v, hmatgrid(:, 1))
          // c = DOT_PRODUCT(v, v)
          // d = b*b-4*a*(c-radius**2)
          double a=0.0, b=0.0, c=0.0;
          for (int i=0; i<3; i++) {
             const double pointk = dh[2][i] * dk;
             const double pointj = pointk + dh[1][i] * dj;
             const double v = pointj - gp[0] * dh[0][i];
             a += dh[0][i] * dh[0][i];
             b += 2.0 * v * dh[0][i];
             c += v * v;
          }
          double d = b * b -4 * a * (c - radius * radius);
          if (d < 0.0) {
             continue;
          }

          // prepare for computing -zetp*rsq
          d = sqrt(d);
          const int ismin = ceill((-b-d)/(2.0*a));
          const int ismax = floor((-b+d)/(2.0*a));
          a *= -zetp;
          b *= -zetp;
          c *= -zetp;
          const int i = ismin - 1;

          // the recursion relation might have to be done
          // from the center of the gaussian (in both directions)
          // instead as the current implementation from an edge
          double exp2i = exp((a * i + b) * i + c);
          double exp1i = exp(2.0 * a * i + a + b);
          const double exp0i = exp(2.0 * a);

          for (int i=ismin; i<=ismax; i++) {
             const double di = i - gp[0];

             // polynomial terms
             double res = 0.0;
             double dip = 1.0;
             for (int il=0; il<=lp; il++) {
                res += coef_xtt[il] * dip;
                dip *= di;
             }

             // the exponential recursion
             exp2i *= exp1i;
             exp1i *= exp0i;
             res *= exp2i;

             int i_index;
             if (periodic[0]) {
                i_index = mod(i, npts[0]) + 1;
             } else {
                i_index = i - index_min[0] + offset[0];
             }
             grid[k_index-1][j_index-1][i_index-1] += res;
          }
       }
    }
}

// *****************************************************************************
static void grid_collocate_internal(const bool use_ortho,
                                    const int func,
                                    const int la_max,
                                    const int la_min,
                                    const int lb_max,
                                    const int lb_min,
                                    const double zeta,
                                    const double zetb,
                                    const double rscale,
                                    const double dh[3][3],
                                    const double dh_inv[3][3],
                                    const double ra[3],
                                    const double rab[3],
                                    const int npts[3],
                                    const int ngrid[3],
                                    const int lb_grid[3],
                                    const bool periodic[3],
                                    const double radius,
                                    const int o1,
                                    const int o2,
                                    const int n1,
                                    const int n2,
                                    const double pab[n2][n1],
                                    double grid[ngrid[2]][ngrid[1]][ngrid[0]]){

    const double zetp = zeta + zetb;
    const double f = zetb / zetp;
    const double rab2 = rab[0] * rab[0] + rab[1] * rab[1] + rab[2] * rab[2];
    const double prefactor = rscale * exp(-zeta * f * rab2);
    double rp[3], rb[3];
    for (int i=0; i<3; i++) {
        rp[i] = ra[i] + f * rab[i];
        rb[i] = ra[i] + rab[i];
    }

    int la_min_diff, la_max_diff, lb_min_diff, lb_max_diff;
    grid_prepare_get_ldiffs(func,
                            &la_min_diff, &la_max_diff,
                            &lb_min_diff, &lb_max_diff);

    const int la_min_prep = max(la_min + la_min_diff, 0);
    const int lb_min_prep = max(lb_min + lb_min_diff, 0);
    const int la_max_prep = la_max + la_max_diff;
    const int lb_max_prep = lb_max + lb_max_diff;

    const int n1_prep = ncoset[la_max_prep];
    const int n2_prep = ncoset[lb_max_prep];
    double pab_prep[n2_prep][n1_prep];
    memset(pab_prep, 0, n2_prep*n1_prep*sizeof(double));

    grid_prepare_pab(func, o1, o2, la_max, la_min, lb_max, lb_min,
                     zeta, zetb, n1, n2, pab, n1_prep, n2_prep, pab_prep);

    //   *** initialise the coefficient matrix, we transform the sum
    //
    // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
    //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya (z-a_z)**lza
    //
    // into
    //
    // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
    //
    // where p is center of the product gaussian, and lp = la_max + lb_max
    // (current implementation is l**7)
    //

    tensor alpha;
    initialize_tensor_4(&alpha, 3, lb_max_prep + 1, la_max_prep + 1, la_max_prep + lb_max_prep + 1);
#ifdef LIBXSMM
    alpha.data = libxsmm_aligned_scratch(sizeof(double) * alpha.alloc_size_, 0/*auto-alignment*/);
#else
    alpha.data = (double*)tmp;
    tmp += alpha.alloc_size_ * sizeof(double);
#endif


    grid_prepare_alpha(ra,
                       rb,
                       rp,
                       la_max_prep,
                       lb_max_prep,
                       &alpha);

    //
    //   compute P_{lxp,lyp,lzp} given P_{lxa,lya,lza,lxb,lyb,lzb} and alpha(ls,lxa,lxb,1)
    //   use a three step procedure
    //   we don't store zeros, so counting is done using lxyz,lxy in order to have
    //   contiguous memory access in collocate_fast.F
    //

    const int lp = la_max_prep + lb_max_prep;
    tensor coef_xyz;
    initialize_tensor_3(&coef_xyz, lp + 1, lp + 1, lp + 1);
#ifdef LIBXSMM
    coef_xyz.data = libxsmm_aligned_scratch(sizeof(double) * coef_xyz.alloc_size_, 0/*auto-alignment*/);
#else
    coef_xyz.data = (double*)tmp;
    tmp += coef_xyz.alloc_size_ * sizeof(double);
#endif


    grid_prepare_coef(la_max_prep,
                      la_min_prep,
                      lb_max_prep,
                      lb_min_prep,
                      lp,
                      prefactor,
                      &alpha,
                      pab_prep,
                      &coef_xyz);

    if (use_ortho) {
        grid_collocate_ortho(lp,
                             zetp,
                             &coef_xyz,
                             dh,
                             dh_inv,
                             rp,
                             npts,
                             lb_grid,
                             periodic,
                             radius,
                             ngrid,
                             grid);
    } else {
        grid_collocate_general(lp,
                               zetp,
                               &coef_xyz,
                               dh,
                               dh_inv,
                               rp,
                               npts,
                               lb_grid,
                               periodic,
                               radius,
                               ngrid,
                               grid);
    }
}


// *****************************************************************************
void grid_collocate_pgf_product_cpu(const bool use_ortho,
                                    const int func,
                                    const int la_max,
                                    const int la_min,
                                    const int lb_max,
                                    const int lb_min,
                                    const double zeta,
                                    const double zetb,
                                    const double rscale,
                                    const double dh[3][3],
                                    const double dh_inv[3][3],
                                    const double ra[3],
                                    const double rab[3],
                                    const int npts[3],
                                    const int ngrid[3],
                                    const int lb_grid[3],
                                    const bool periodic[3],
                                    const double radius,
                                    const int o1,
                                    const int o2,
                                    const int n1,
                                    const int n2,
                                    const double pab[n2][n1],
                                    double grid[ngrid[2]][ngrid[1]][ngrid[0]]){

// Uncomment this to dump all tasks to file.
// #define __GRID_DUMP_TASKS

#ifdef __GRID_DUMP_TASKS
    // Can be large, run with "ulimit -s unlimited".
    double grid_before[ngrid[2]][ngrid[1]][ngrid[0]];
    for (int i=0; i<ngrid[2]; i++) {
    for (int j=0; j<ngrid[1]; j++) {
    for (int k=0; j<ngrid[0]; j++) {
        grid_before[i][j][k] = grid[i][j][k];
        grid[i][j][k] = 0.0;
    }
    }
    }
#endif

    grid_collocate_internal(use_ortho,
                            func,
                            la_max,
                            la_min,
                            lb_max,
                            lb_min,
                            zeta,
                            zetb,
                            rscale,
                            dh,
                            dh_inv,
                            ra,
                            rab,
                            npts,
                            ngrid,
                            lb_grid,
                            periodic,
                            radius,
                            o1,
                            o2,
                            n1,
                            n2,
                            pab,
                            grid);

#ifdef __GRID_DUMP_TASKS

    grid_collocate_record(use_ortho,
                          func,
                          la_max,
                          la_min,
                          lb_max,
                          lb_min,
                          zeta,
                          zetb,
                          rscale,
                          dh,
                          dh_inv,
                          ra,
                          rab,
                          npts,
                          ngrid,
                          lb_grid,
                          periodic,
                          radius,
                          o1,
                          o2,
                          n1,
                          n2,
                          pab,
                          grid);

    for (int i=0; i<ngrid[2]; i++) {
        for (int j=0; j<ngrid[1]; j++) {
            for (int k=0; j<ngrid[0]; j++) {
                grid[i][j][k] += grid_before[i][j][k];
            }
        }
    }
#endif

}

//EOF
