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
extern void apply_non_orthorombic_corrections_xy(const int x, const int y, const struct tensor_ *const Exp, struct tensor_ *const m);
extern void apply_non_orthorombic_corrections_xz(const int x, const int z, const struct tensor_ *const Exp, struct tensor_ *const m);
extern void apply_non_orthorombic_corrections_yz(const int y, const int z, const struct tensor_ *const Exp, struct tensor_ *const m);

#endif
