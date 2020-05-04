#include "tensor_local.h"
#include "collocation_integration.h"
#include "utils.h"

extern void collocate_core_rectangular(double *scratch,
                                       const double prefactor,
                                       const struct tensor_ *co,
                                       const struct tensor_ *p_alpha_beta_reduced_,
                                       struct tensor_ *cube);

/*
 * create a collocation list of a fixed number of elements
 * first parameter : number of elements in the list
 * second parameter : do we integrate or collocate. For integration a local copy
 * of the grid might be stored for convenience
 */
collocation_list *create_collocation_list(const int num_elem, const int integration)
{
    struct collocation_list_ *list = (struct collocation_list_ *) malloc(sizeof(struct collocation_list_));
    memset(list, 0, sizeof(struct collocation_list_));

    list->total_number_of_elements_ = num_elem;

    list->list = (struct collocation_block_ *) malloc(sizeof(struct collocation_block_) * list->total_number_of_elements_);
    memset(list->list, 0, sizeof(struct collocation_block_) * list->total_number_of_elements_);

    if (!list->list)
        abort();

    list->first_round = true;
    list->scratch = NULL;
    list->integration = integration;
    return list;
}

void destroy_collocation_list(struct collocation_list_ *list_collocation)
{
    if (list_collocation != NULL) {
        free(list_collocation->list);
        free(list_collocation->scratch);
        free(list_collocation);
    }
}

void print_collocation_block(const struct collocation_block_ *const list)
{
    if (!list)
        abort();
    printf("-------------------------\n\n");
    printf("pol address  : %p\n\n", (void*)list->pol.data);

    printf("-------------------------\n\n");
    printf("Cube address : %p\n", list->cube.data);
    printf("Cube offset  : %d %d %d\n", list->position_inside_cube[0], list->position_inside_cube[1], list->position_inside_cube[2]);
    printf("cube size    : %d %d %d\n", list->cube.size[0], list->cube.size[1], list->cube.size[2]);
    printf("-------------------------\n\n");
    printf("grid info\n");
    printf("grid size    : %d %d %d\n", list->grid.size[0], list->grid.size[1], list->grid.size[2]);
    printf("lower corner : %d %d %d\n", list->lower_corner[0], list->lower_corner[1], list->lower_corner[2]);
    printf("upper corner : %d %d %d\n", list->upper_corner[0], list->upper_corner[1], list->upper_corner[2]);
}

void *print_collocation_list(void *const handler)
{
    struct collocation_list_ *list = (struct collocation_list_*)handler;

    if (list->number_of_elements_ == 0)
        return NULL;

    if (list->done)
        return NULL;

    printf("list block              %p\n", list);
    printf("list number of elements %d\n", list->number_of_elements_);
    for (int l = 0; l < list->number_of_elements_; l++){
        printf("block %d\n", l);
        print_collocation_block(list->list + l);
    }

    return NULL;
}

void mark_collocation_new_pair(struct collocation_integration_ *const handler)
{
    if(handler == NULL) {
        abort();
    }


    struct collocation_list_ *list_collocation = handler->current_list;

    if (handler->current_list == NULL) {
        list_collocation = handler->list[0];
    }

    if (list_collocation->number_of_elements_ < list_collocation->total_number_of_elements_) {
        struct collocation_block_ *list = &list_collocation->list[list_collocation->number_of_elements_];
        list->new_gaussian_pair_ = true;
    }
}

void add_collocation_block(struct collocation_integration_ *const handler,
                           const int *period,
                           const int *cube_size,
                           const int *position_inside_cube,
                           const int *lower_corner,
                           const int *upper_corner,
                           const tensor *Exp,
                           tensor *grid)
{
    int position1[3];
    if (position_inside_cube) {
        position1[0] = position_inside_cube[0];
        position1[1] = position_inside_cube[1];
        position1[2] = position_inside_cube[2];
    } else {
        position1[0] = 0;
        position1[1] = 0;
        position1[2] = 0;
    }

    struct collocation_list_ *list_collocation = handler->current_list;

    if (list_collocation == NULL) {
        /* It means that the two queues are empty */
        handler->current_list = handler->list[0];
        handler->working_thread = 0;
        list_collocation = handler->current_list;
        handler->list[0]->first_round = false;
        handler->list[1]->first_round = true;
        handler->list[1]->cube_alloc_size = 0;
        handler->list[0]->cube_alloc_size = 0;
    }

    if (list_collocation->number_of_elements_ == list_collocation->total_number_of_elements_) {
        /* the list of blocks is full. Two steps
         *
         * 1) wait for the thread pool to finish currently processed jobs. If
         * the list is small for instance, it might take more time to process is
         * than to create it. So we have to wait before switching

         * 2) start the new one
         * 3) swap the working lists
         */

        /* step one : wait */
        if (!handler->sequential_mode) {
            thpool_wait(handler->thpool);

            /* do not permute the two lines because otherwise the code waits for the */
            /*        two queues to finish */

            /*            step two */
            /* allocate new table for the coefficients because the memory is
             * freed inside calculate_collocation. So we can have an invalid
             * pointer sometimes when the blocks of a given gaussian set are
             * distributed over two lists. Basically it is a corner case */
            thpool_add_work(handler->thpool, (void*)calculate_collocation, (void*)list_collocation);
        } else {
            calculate_collocation(list_collocation);
        }

        handler->working_thread++;

        /* we only have two lists */
        if (list_collocation == handler->list[0]) {
            /* work is done in the second list. We can use it for storing the block */
            handler->current_list = handler->list[1];
            handler->list[1]->done = false;
            handler->list[1]->first_round = false;
            handler->list[1]->list[0].new_gaussian_pair_ = true;
        } else {
            /* work is done in the first list. We can use it for storing the block */
            handler->current_list = handler->list[0];
            handler->list[0]->done = false;
            handler->list[0]->first_round = false;
            /* the first element is indicated as a new gaussian pair. It might
             * involve recomputing the cube but it avoids interference between
             * the two lists */
            handler->list[0]->list[0].new_gaussian_pair_ = true;
            handler->list[0]->T_alloc_size = 0;
            handler->list[0]->W_alloc_size = 0;
        }

        /* again do not permute this. It should be last */
        list_collocation = handler->current_list;
        list_collocation->cube_alloc_size = 0;
    }

    struct collocation_block_ *list = &list_collocation->list[list_collocation->number_of_elements_];

    list->period[0] = period[0];
    list->period[1] = period[1];
    list->period[2] = period[2];

/* starting point for the polynomials */
    list->position_inside_cube[0] = position1[0];
    list->position_inside_cube[1] = position1[1];
    list->position_inside_cube[2] = position1[2];

    list->lower_corner[0] = lower_corner[0];
    list->lower_corner[1] = lower_corner[1];
    list->lower_corner[2] = lower_corner[2];

    list->upper_corner[0] = upper_corner[0];
    list->upper_corner[1] = upper_corner[1];
    list->upper_corner[2] = upper_corner[2];

    list->Exp.data = NULL;
    list->pol.data = NULL;
    list->cube.data = NULL;
    list->coefs.data = NULL;
    /* I do not allocate memory yet. It will be done when we do the actual calculations */
    initialize_tensor_3(&list->cube,
                        cube_size[0],
                        cube_size[1],
                        cube_size[2]);

    /* compute the size of the largest cube */
    list_collocation->cube_alloc_size = max(list_collocation->cube_alloc_size, list->cube.alloc_size_);

    if (list->new_gaussian_pair_) {
        initialize_tensor_3(&list->coefs, handler->coef.size[0], handler->coef.size[1], handler->coef.size[2]);

        posix_memalign((void **)&list->coefs.data, 64, sizeof(double) * handler->coef.alloc_size_);
        memcpy(list->coefs.data, handler->coef.data, sizeof(double) * handler->coef.alloc_size_);


        if (Exp != NULL) {
            initialize_tensor_3(&list->Exp, Exp->size[0], Exp->size[1], Exp->size[2]);
            posix_memalign((void **)&list->Exp.data, 64, sizeof(double) * Exp->alloc_size_);
            memcpy(list->Exp.data, Exp->data, sizeof(double) * Exp->alloc_size_);
        }

        initialize_tensor_3(&list->pol,
                            handler->pol.size[0],
                            handler->pol.size[1],
                            handler->pol.size[2]);
        posix_memalign((void **)&list->pol.data, 64, sizeof(double) * handler->pol.alloc_size_);
        memcpy(list->pol.data, handler->pol.data, sizeof(double) * handler->pol.alloc_size_);

        /* It is for the second variant of collocate_dgemm only three dgemm */
        list_collocation->T_alloc_size = max(list_collocation->T_alloc_size,
                                             compute_memory_space_tensor_3(handler->coef.size[0] /* alpha */,
                                                                           handler->coef.size[1] /* gamma */,
                                                                           cube_size[1] /* j */));

        list_collocation->W_alloc_size = max(list_collocation->W_alloc_size,
                                             compute_memory_space_tensor_3(handler->coef.size[1] /* gamma */ ,
                                                                           cube_size[1] /* j */,
                                                                           cube_size[2] /* i */));

        handler->W_alloc_size = max(handler->W_alloc_size, list_collocation->W_alloc_size);
        handler->T_alloc_size = max(handler->T_alloc_size, list_collocation->T_alloc_size);
    }

    /* we could put that in handle ? */
    initialize_tensor_3(&list->grid, grid->size[0], grid->size[1], grid->size[2]);
    list->grid.data = grid->data;
    list->initialized_ = 1;
    list_collocation->number_of_elements_++;
}

void *collocate_create_handle(const int device_id, const int number_of_gaussian, const bool sequential_mode)
{
    struct collocation_integration_ *handle = NULL;
    handle = (struct collocation_integration_ *) malloc(sizeof(struct collocation_integration_));

    if (handle == NULL) {
        abort();
    }

    memset(handle, 0, sizeof(struct collocation_integration_));

    handle->gpu_id = device_id;
    handle->sequential_mode = sequential_mode;

    if (!handle->sequential_mode)
        handle->thpool = thpool_init(1);

    handle->list[0] = create_collocation_list(number_of_gaussian, 0);
    handle->list[1] = create_collocation_list(number_of_gaussian, 0);
    handle->number_of_gaussian = number_of_gaussian;
    handle->list[0]->initialized_ = 0;
    handle->list[1]->initialized_ = 0;
    handle->list[0]->done = 0;
    handle->list[1]->done = 0;
    handle->list[0]->number_of_elements_ = 0;
    handle->list[1]->number_of_elements_ = 0;

    if (sequential_mode) {
        handle->alpha.alloc_size_ = 8192;
        handle->coef.alloc_size_ = 1024;
        handle->pol.alloc_size_ = 1024;
        /* it is a cube of size 32 x 32 x 32 */
        handle->cube.alloc_size_ = 32768;

        handle->cube_alloc_size = realloc_tensor(&handle->cube);
        handle->alpha_alloc_size = realloc_tensor(&handle->alpha);
        handle->coef_alloc_size = realloc_tensor(&handle->coef);
        handle->pol_alloc_size = realloc_tensor(&handle->pol);

        posix_memalign((void**)&handle->scratch, 32, sizeof(double) * 10240);
        handle->scratch_alloc_size = 10240;
        handle->T_alloc_size = 8192;
        handle->W_alloc_size = 2048;
        handle->blockDim[0] = 5;
        handle->blockDim[1] = 5;
        handle->blockDim[2] = 5;
    }

    return (void*)handle;
}

void collocate_synchronize(void *gaussian_handler)
{
    if (gaussian_handler == NULL) {
        abort();
    }

    struct collocation_integration_ *handler = (struct collocation_integration_ *)gaussian_handler;
    if ((handler->sequential_mode) && (!handler->grid_restored)) {
        if (handler->blocked_grid.blocked_decomposition) {
            add_blocked_tensor_to_tensor(&handler->blocked_grid, &handler->grid);
            memset(handler->blocked_grid.data, 0, sizeof(double) * handler->blocked_grid.alloc_size_);
            handler->grid_restored = true;
        } else {
            return;
        }
    }
    if (!handler->sequential_mode)
    {
        thpool_wait(handler->thpool);

        if ((handler->list[0]->number_of_elements_ > 0) && (!handler->list[0]->done)) {
            calculate_collocation(handler->list[0]);
        }

        if ((handler->list[1]->number_of_elements_ > 0) && (!handler->list[1]->done)) {
            calculate_collocation(handler->list[1]);
        }
    }

}

void collocate_finalize(void *gaussian_handle)
{
    collocate_synchronize(gaussian_handle);
    struct collocation_integration_ *handle = (struct collocation_integration_ *)gaussian_handle;

    destroy_collocation_list(handle->list[0]);
    destroy_collocation_list(handle->list[1]);

    free(handle->alpha.data);
    free(handle->coef.data);

    /* release the thread pool */
    if (!handle->sequential_mode)
        thpool_destroy(handle->thpool);

    if (handle->Exp.data)
        free(handle->Exp.data);

    free(handle->scratch);
    free(handle->pol.data);
    free(handle->cube.data);

    handle->alpha.data = NULL;
    handle->coef.data = NULL;


    /* free(handle->grid_test.data); */

    free(handle);

    handle = NULL;
}

void calculate_collocation(void *const in)
{
    struct collocation_list_ *const list_ = (struct collocation_list_ *)in;

    if (list_->scratch == NULL) {
        list_->scratch_size = list_->cube_alloc_size + list_->T_alloc_size + list_->W_alloc_size;
        posix_memalign((void **)&list_->scratch, 64, sizeof(double) * list_->scratch_size);
    }

    if (list_->scratch_size < (list_->cube_alloc_size + list_->T_alloc_size + list_->W_alloc_size))
    {
        list_->scratch_size = list_->cube_alloc_size + list_->T_alloc_size + list_->W_alloc_size;
        free(list_->scratch);
        posix_memalign((void **)&list_->scratch, 64, sizeof(double) * list_->scratch_size);
    }

    for (int i = 0; i < list_->number_of_elements_; i++) {

        /* print_collocation_block(list_->list + i); */

        list_->list[i].cube.data = list_->scratch;

        if (list_->list[i].new_gaussian_pair_) {
            tensor_reduction_for_collocate_integrate(list_->scratch + list_->list[i].cube.alloc_size_,
                                                     1.0,
                                                     &list_->list[i].coefs,
                                                     &list_->list[i].pol,
                                                     &list_->list[i].cube);

            free(list_->list[i].pol.data);
            free(list_->list[i].coefs.data);
            if (list_->list[i].Exp.data) {

                // Well self explanatory
                apply_non_orthorombic_corrections(list_->list[i].plane,
                                                  &list_->list[i].Exp,
                                                  &list_->list[i].cube);

                free(list_->list[i].Exp.data);
            }
            list_->list[i].Exp.data = NULL;
            list_->list[i].pol.data = NULL;
            list_->list[i].coefs.data = NULL;
        }

        add_sub_grid(list_->list[i].lower_corner, // lower corner position where the subgrid should placed
                     list_->list[i].upper_corner, // upper boundary
                     list_->list[i].position_inside_cube, // starting position of in the subgrid
                     &list_->list[i].cube, // subgrid
                     &list_->list[i].grid);

        // I forgot that one !!!!!!!
        list_->list[i].new_gaussian_pair_ = false;
    }

    list_->done = true;
    list_->number_of_elements_ = 0;
    return;
}


/* compute the decomposition of the cube according to the boundaries conditions
 * and create a list of small tasks with the size and position of the subblock
 * to be added or extracted from/to the grid. It does not modify the grid what
 * so ever */
void compute_blocks(collocation_integration *const handler,
                    const int *lower_boundaries_cube,
                    const int *cube_size,
                    const int *cube_center,
                    const int *period,
                    const tensor *Exp,
                    const int *lb_grid,
                    tensor *grid)
{
    int position[3];
    return_cube_position(grid->size, lb_grid, cube_center, lower_boundaries_cube, period, position);
    mark_collocation_new_pair(handler);
    if ((position[1] + cube_size[1] <= grid->size[1]) &&
        (position[2] + cube_size[2] <= grid->size[2]) &&
        (position[0] + cube_size[0] <= grid->size[0])) {
        // it means that the cube is completely inside the grid without touching
        // the grid borders. periodic boundaries conditions are pointless here.
        // we can simply loop over all three dimensions.

        // it also consider the case where they are open boundaries

        const int upper_corner[3] = {position[0] + cube_size[0],
                                     position[1] + cube_size[1],
                                     position[2] + cube_size[2]};

        add_collocation_block(handler,
                              period,
                              cube_size,
                              NULL,
                              position,
                              upper_corner,
                              Exp,
                              grid);
        return;
    }

    int z1 = position[0];
    int z_offset = 0;
    /* We actually split the cube into smaller parts such that we do not have
     * to apply pcb as a last stage. The blocking takes care of it */

    int lower_corner[3];
    int upper_corner[3];

    for (int z = 0; (z < (cube_size[0] - 1)); z++, z1++) {
        lower_corner[0] = z1;
        upper_corner[0] = compute_next_boundaries(&z1, z, grid->size[0], period[0], cube_size[0]);

        /* // We have a full plane. */

        if (upper_corner[0] - lower_corner[0] > 0) {
            int y1 = position[1];
            int y_offset = 0;
            for (int y = 0; y < cube_size[1]; y++, y1++) {
                lower_corner[1] = y1;
                upper_corner[1] = compute_next_boundaries(&y1, y, grid->size[1], period[1], cube_size[1]);

                /*     // this is needed when the grid is distributed over several ranks. */
                /* if (y1 >= lb_grid[1] + grid->size[1]) */
                /*     continue; */

                if (upper_corner[1] - lower_corner[1] > 0) {
                    int x1 = position[2];
                    int x_offset = 0;

                    for (int x = 0; x < cube_size[2]; x++, x1++) {
                        lower_corner[2] = x1;
                        upper_corner[2] = compute_next_boundaries(&x1, x, grid->size[2], period[2], cube_size[2]);
                        if (upper_corner[2] - lower_corner[2] > 0) {

                            int position2[3]= {z_offset, y_offset, x_offset};

                            add_collocation_block(handler,
                                                  period,
                                                  cube_size,
                                                  position2, // starting position in the subgrid
                                                  lower_corner,
                                                  upper_corner,
                                                  Exp,
                                                  grid);

                        }

                        update_loop_index(lower_corner[2], upper_corner[2], grid->size[2], period[2], &x_offset, &x, &x1);
                    }
                    /* this dimension of the grid is divided over several ranks */
                }
                update_loop_index(lower_corner[1], upper_corner[1], grid->size[1], period[1], &y_offset, &y, &y1);
            }
        }
        update_loop_index(lower_corner[0], upper_corner[0], grid->size[0], period[0], &z_offset, &z, &z1);
    }
}

void initialize_W_and_T(collocation_integration *const handler, const tensor *cube, const tensor *coef)
{
    size_t tmp1 = compute_memory_space_tensor_3(coef->size[0] /* alpha */,
                                                coef->size[1] /* gamma */,
                                                cube->size[1] /* j */);

    size_t tmp2 = compute_memory_space_tensor_3(coef->size[0] /* gamma */ ,
                                                cube->size[1] /* j */,
                                                cube->size[2] /* i */);

    const size_t mem_alloc_size_ = max(max(tmp1 + tmp2, cube->alloc_size_), coef->alloc_size_);
    if ((mem_alloc_size_ > handler->scratch_alloc_size) || (handler->scratch == NULL)) {
        handler->T_alloc_size = tmp1;
        handler->W_alloc_size = tmp2;

        handler->scratch_alloc_size = mem_alloc_size_;

        if (handler->scratch)
            free(handler->scratch);
        if (posix_memalign(&handler->scratch, 64, sizeof(double) * handler->scratch_alloc_size) != 0)
            abort();
    }
}
