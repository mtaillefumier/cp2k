#ifndef _NON_ORTHO_CORRECTIONS_H
#define _NON_ORTHO_CORRECTIONS_H

extern void calculate_non_orthorombic_corrections_tensor(const double mu_mean,
                                                         const double *r_ab,
                                                         const double basis[3][3],
                                                         const int *const xmin,
                                                         const int *const xmax,
                                                         bool *plane,
                                                         tensor *const Exp);
extern void apply_non_orthorombic_corrections(const bool *__restrict plane,
                                              const tensor *const Exp,
                                              tensor *const cube);

#endif
