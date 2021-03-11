#ifndef TASK_HPP
#define TASK_HPP

extern "C" {
#include "../common/grid_basis_set.h"
}
struct task_info {
    int level;
    int iatom;
    int jatom;
    int iset;
    int jset;
    int ipgf;
    int jpgf;
    int ikind;
    int jkind;
    int sgfa;
    int sgfb;
    int nsgfa;
    int nsgfb;
    int nsgf_seta;
    int nsgf_setb;
    int ncoseta;
    int ncosetb;
    int maxcoa;
    int maxcob;
    int first_coseta;
    int first_cosetb;
    int subblock_offset;
    grid_basis_set *ibasis;
    grid_basis_set *jbasis;
    int border_mask;
    int block_num;
    double radius;
    double zetp;
    double zeta[2];
    double ra[3];
    double rb[3];
    double rp[3];
    int lmax[2];
    int lmin[2];
    int l1_plus_l2_;
    int offset[2];
    bool update_block_;
    double rab[3];
    double prefactor;
};
#endif
