#ifndef TASK_LIST_H
#define TASK_LIST_H

#include <stdbool.h>

#include "grid_common.h"
#include "grid_basis_set.h"

/*******************************************************************************
 * \brief Internal representation of a task.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
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

/*******************************************************************************
 * \brief Internal representation of a task list.
 * \author Ole Schuett
 ******************************************************************************/
typedef struct {
    int ntasks;
    int nlevels;
    int natoms;
    int nkinds;
    int nblocks;
    int buffer_size;
    double *blocks_buffer;
    int *block_offsets;
    double *atom_positions;
    int *atom_kinds;
    grid_basis_set **basis_sets;
    grid_ref_task *tasks;
    int *tasks_per_level;
    int maxco;
    grid_info *grid;
} grid_task_list;

inline int return_grid_local_size(grid_info *const grid){
    return grid->grid_local_size[0] * grid->grid_local_size[1] * grid->grid_local_size[2];
}

#endif
