#ifndef GRID_COLLOCATE_GPU__
#define GRID_COLLOCATE_GPU__
#include "../common/grid_tasklist_private.h"

void
grid_collocate_task_list_gpu(const int device_id, const grid_task_list_private* task_list, const bool orthorhombic, const int func,
                             const int nlevels, const int npts_global[nlevels][3], const int npts_local[nlevels][3],
                             const int shift_local[nlevels][3], const int border_width[nlevels][3],
                             const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3], double* grid[nlevels]);
#endif
