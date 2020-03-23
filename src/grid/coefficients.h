#ifndef COEFFICIENTS_H
#define COEFFICIENTS_H

// *****************************************************************************
static void grid_prepare_coef_ortho(const int *lmax,
                                    const int *lmin,
                                    const int lp,
                                    const double prefactor,
                                    const tensor *alpha, // [3][lb_max+1][la_max+1][lp+1]
                                    const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                                    tensor *coef_xyz) //[lp+1][lp+1][lp+1]
{


    memset(coef_xyz->data, 0, coef_xyz->alloc_size_ * sizeof(double));

    // we need a proper fix for that. We can use the tensor structure for this

    double coef_xyt[lp+1][lp+1];
    double coef_xtt[lp+1];

    for (int lzb = 0; lzb<=lmax[1]; lzb++) {
        for (int lza = 0; lza<=lmax[0]; lza++) {
            for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                    coef_xyt[lyp][lxp] = 0.0;
                }
            }
            for (int lyb = 0; lyb<=lmax[1]-lzb; lyb++) {
                for (int lya = 0; lya<=lmax[0]-lza; lya++) {
                    const int lxpm = (lmax[1]-lzb-lyb) + (lmax[0]-lza-lya);
                    for (int i=0; i<=lxpm; i++) {
                        coef_xtt[i] = 0.0;
                    }
                    for (int lxb = max(lmin[1]-lzb-lyb, 0); lxb<=lmax[1]-lzb-lyb; lxb++) {
                        for (int lxa = max(lmin[0]-lza-lya, 0); lxa<=lmax[0]-lza-lya; lxa++) {
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
                        idx3(coef_xyz[0], lxp, lzp, lyp) += idx4(alpha[0], 2, lzb, lza, lzp) * coef_xyt[lyp][lxp];
                    }
                }
            }
        }
    }
}

// *****************************************************************************
static void grid_prepare_alpha(const double ra[3],
                               const double rb[3],
                               const double rp[3],
                               const int *lmax,
                               tensor *alpha)
{
    // Initialize with zeros.
    memset(alpha->data, 0, alpha->alloc_size_ * sizeof(double));

    //
    //   compute polynomial expansion coefs -> (x-a)**lxa (x-b)**lxb -> sum_{ls} alpha(ls,lxa,lxb,1)*(x-p)**ls
    //

    for (int iaxis=0; iaxis<3; iaxis++) {
        const double drpa = rp[iaxis] - ra[iaxis];
        const double drpb = rp[iaxis] - rb[iaxis];
        for (int lxa = 0; lxa <= lmax[0]; lxa++) {
            for (int lxb = 0; lxb <= lmax[1]; lxb++) {
                double binomial_k_lxa = 1.0;
                double a = 1.0;
                for (int k = 0; k <= lxa; k++) {
                    double binomial_l_lxb = 1.0;
                    double b = 1.0;
                    for (int l = 0; l <= lxb; l++) {
                        idx4(alpha[0], iaxis, lxb, lxa, lxa - l + lxb - k) += binomial_k_lxa * binomial_l_lxb * a * b;
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
static void grid_prepare_coef(const int *lmax,
                              const int *lmin,
                              const int lp,
                              const double prefactor,
                              const tensor *alpha,
                              const double pab[ncoset[lmax[1]]][ncoset[lmax[0]]],
                              tensor *coef_xyz)
{

    memset(coef_xyz->data, 0, coef_xyz->alloc_size_ * sizeof(double));

    // we need a proper fix for that. We can use the tensor structure for this

    double coef_xyt[lp+1][lp+1];
    double coef_xtt[lp+1];

    for (int lzb = 0; lzb<=lmax[1]; lzb++) {
        for (int lza = 0; lza<=lmax[0]; lza++) {
            for (int lyp = 0; lyp<=lp-lza-lzb; lyp++) {
                for (int lxp = 0; lxp<=lp-lza-lzb-lyp; lxp++) {
                    coef_xyt[lyp][lxp] = 0.0;
                }
            }
            for (int lyb = 0; lyb<=lmax[1]-lzb; lyb++) {
                for (int lya = 0; lya<=lmax[0]-lza; lya++) {
                    const int lxpm = (lmax[1]-lzb-lyb) + (lmax[0]-lza-lya);
                    for (int i=0; i<=lxpm; i++) {
                        coef_xtt[i] = 0.0;
                    }
                    for (int lxb = max(lmin[1]-lzb-lyb, 0); lxb<=lmax[1]-lzb-lyb; lxb++) {
                        for (int lxa = max(lmin[0]-lza-lya, 0); lxa<=lmax[0]-lza-lya; lxa++) {
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


#endif
