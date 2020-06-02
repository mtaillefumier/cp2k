/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2020  CP2K developers group                         *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <mkl.h>

#include "grid_collocate_replay.h"

int
main(int argc, char* argv[])
{
    if (argc != 5) {
        printf("Usage: grid_base_ref_miniapp.x <task-file> num_cycles num_blocks 0 or 1\n");
        return 1;
    }
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    const int cycles      = atoi(argv[2]); // For better statistics the task is collocated many times.
    const double max_diff = grid_collocate_replay(argv[1], cycles, atoi(argv[3]), !atoi(argv[4]));
    assert(max_diff < 1e-11 * cycles);
    return 0;
}

// EOF
