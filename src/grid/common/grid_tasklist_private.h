#ifndef _TASK_LIST_PRIVATE_H
#define _TASK_LIST_PRIVATE_H

#include "grid_basis_set.h"

//******************************************************************************
// \brief Internal representation of a task.
// \author Ole Schuett
//******************************************************************************
typedef struct
{
    int level;
    int iatom;
    int jatom;
    int iset;
    int jset;
    int ipgf;
    int jpgf;
    int border_mask;
    int block_num;
    double radius;
    double rab[3];
} grid_task;

//******************************************************************************
// \brief Internal representation of a task list.
// \author Ole Schuett
//******************************************************************************
typedef struct
{
    int ntasks;
    int nlevels;
    int natoms;
    int nkinds;
    int nblocks;
    int buffer_size;
    double* blocks_buffer;
    int* block_offsets;
    double* atom_positions;
    int* atom_kinds;
    grid_basis_set** basis_sets;
    grid_task* tasks;
    int* tasks_per_level;
    int maxco;
} grid_task_list_private;

#endif
