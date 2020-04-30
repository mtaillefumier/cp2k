#ifndef __COLLOCATION_INTEGRATION_H
#define __COLLOCATION_INTEGRATION_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tensor_local.h"
#include "thpool.h"

typedef struct collocation_block_ {
    /* */
    bool plane[3];
    tensor pol;
    tensor coefs;
    tensor Exp;
    tensor cube;
    tensor grid;

    /* intermdiate storage for the collocation. We don not allocate these in
     * advance because it is method dependent */

    tensor T;
    tensor W;
    int position_inside_cube[3];
    int lower_corner[3];
    int upper_corner[3];
    int period[3];
    int initialized_;
    /* first block indicating a new gaussian pair */
    /* this mean that the full cube will be computed. */
    /* the next blocks will only store the positions etc... */

    bool new_gaussian_pair_;
} collocation_block;

typedef struct collocation_list_ {
    struct collocation_block_ *list;
    int number_of_elements_;
    int total_number_of_elements_;
    bool done;
    bool first_round;
    int initialized_;
    double *scratch;
    size_t scratch_size;
    size_t cube_alloc_size;
    size_t coef_alloc_size;
    size_t alpha_alloc_size;
    size_t pol_alloc_size;
    size_t T_alloc_size;
    size_t W_alloc_size;
    int integration;
} collocation_list;

typedef struct collocation_integration_ {
/* GPU device id. should replace this with GPU UID */
    int gpu_id;

    /*
      Do we want the batched or the serial mode. The difference between the two
      modes is that in one case group of gaussians are treated collectively
      while they are treated one by one in the other case. When the GPU mode is
      activated then the batch mode is also activated.
    */
    bool sequential_mode;

    /* two lists containing information about the computations to do. Only
     * allocated when the serial is off */
    struct collocation_list_ *list[2];

    /* current list to be filled */
    struct collocation_list_ *current_list;

    /* Just for debugging */
    int working_thread;

    /* number of gaussians block in each list */
    int number_of_gaussian;

    /* structure for handling the thread pool */
    threadpool thpool;

    /* some scratch storage to avoid malloc / free all the time */
    tensor alpha;
    tensor pol;
    tensor coef;

    /* tensors for the grid to collocate or integrate */
    /* original grid */
    tensor grid;
    /* original grid decomposed in block */
    tensor blocked_grid;

    /* do we need to update the grid */
    bool grid_restored;

    /* block dimensions */
    int blockDim[4];

/* Only allocated in sequential mode */
    tensor cube;
    tensor Exp;
    bool plane[3];
    size_t Exp_alloc_size;
    size_t cube_alloc_size;
    size_t coef_alloc_size;
    size_t alpha_alloc_size;
    size_t pol_alloc_size;
    size_t scratch_alloc_size;
    size_t T_alloc_size;
    size_t W_alloc_size;

    void *scratch;
} collocation_integration;


extern collocation_list *create_collocation_list(const int num_elem, const int integration);
extern void destroy_collocation_list(struct collocation_list_ *list_collocation);
extern void print_collocation_block(const struct collocation_block_ *const list);
extern void *print_collocation_list(void *const handler);
extern void mark_collocation_new_pair(struct collocation_integration_ *const handler);
extern void add_collocation_block(struct collocation_integration_ *const handler,
                                  const int *period,
                                  const int *cube_size,
                                  const int *position_inside_cube,
                                  const int *lower_corner,
                                  const int *upper_corner,
                                  const tensor *Exp,
                                  tensor *grid);
extern void *collocate_create_handle(const int device_id, const int number_of_gaussian, const bool sequential_mode);
extern void collocate_synchronize(void *gaussian_handler);
extern void collocate_finalize(void *gaussian_handle);
extern void calculate_collocation(void *const in);
extern void compute_blocks(collocation_integration *const handler,
                           const int *const lower_boundaries_cube,
                           const int *const cube_size,
                           const int *const cube_center,
                           const int *const period,
                           const tensor *const Exp,
                           const int *const lb_grid,
                           tensor *grid);
extern void initialize_W_and_T(collocation_integration *const handler, const tensor *cube, const tensor *coef);
#endif
