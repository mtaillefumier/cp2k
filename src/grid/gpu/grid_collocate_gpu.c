#ifdef __COLLOCATE_GPU

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../common/utils.h"
#include "../common/grid_tasklist_private.h"
#include "../cpu/collocation_integration.h"
#include "../cpu/private_header.h"
#include "../cpu/coefficients.h"
#include "../cpu/grid_prepare_pab_dgemm.h"

void compute_collocation_gpu(pgf_list_gpu* handler);

void initialize_grid_parameters_on_gpu(collocation_integration* handler);

pgf_list_gpu* create_worker_list(const int number_of_workers, const int batch_size, const int num_devices,
                                 const int* device_id);

void destroy_worker_list(pgf_list_gpu* const list);

void
reset_list_gpu(pgf_list_gpu* const list)
{
    cudaSetDevice(list->device_id);
    list->list_length                  = 0;
    list->coef_dynamic_alloc_size_gpu_ = 0;
}

void
my_worker_is_running(pgf_list_gpu* const my_worker)
{
    if (my_worker->running) {
        cudaSetDevice(my_worker->device_id);
        cudaEventSynchronize(my_worker->event);
        reset_list_gpu(my_worker);
        my_worker->running = false;
    }
}

void
add_orbital_to_list(pgf_list_gpu* const list, const int lp, const double rp[3], const double radius, const double zetp,
                    const tensor* const coef)
{
    assert(list->batch_size > list->list_length);

    list->lmax_cpu_[list->list_length] = lp;
    /* list->cube_size_cpu_[3 * list->list_length]     = cube_size[0]; */
    /* list->cube_size_cpu_[3 * list->list_length + 1] = cube_size[1]; */
    /* list->cube_size_cpu_[3 * list->list_length + 2] = cube_size[2]; */

    /* list->cube_position_cpu_[3 * list->list_length]     = cube_position[0]; */
    /* list->cube_position_cpu_[3 * list->list_length + 1] = cube_position[1]; */
    /* list->cube_position_cpu_[3 * list->list_length + 2] = cube_position[2]; */

    list->rp_cpu_[list->list_length].x = rp[0];
    list->rp_cpu_[list->list_length].y = rp[1];
    list->rp_cpu_[list->list_length].z = rp[2];

    list->radius_cpu_[list->list_length] = radius;

    list->zeta_cpu_[list->list_length] = zetp;
    list->coef_offset_cpu_[0]          = 0;
    if (list->list_length > 0) {
        list->coef_offset_cpu_[list->list_length] =
            list->coef_offset_cpu_[list->list_length - 1] + list->coef_previous_alloc_size_;
        list->coef_dynamic_alloc_size_gpu_ += coef->alloc_size_;
    } else {
        list->coef_dynamic_alloc_size_gpu_ = coef->alloc_size_;
    }
    list->coef_previous_alloc_size_ = coef->alloc_size_;

    if (list->coef_dynamic_alloc_size_gpu_ > list->coef_alloc_size_gpu_) {
        list->durty     = true;
        list->coef_cpu_ = realloc(list->coef_cpu_, sizeof(double) * (list->coef_dynamic_alloc_size_gpu_ + 4096));
        list->coef_alloc_size_gpu_ = list->coef_dynamic_alloc_size_gpu_ + 4096;
    }

    if (list->lmax < lp)
        list->lmax = lp;
    memcpy(list->coef_cpu_ + list->coef_offset_cpu_[list->list_length], coef->data, sizeof(double) * coef->alloc_size_);
    list->list_length++;
}

pgf_list_gpu*
create_worker_list(const int number_of_workers, const int batch_size, const int num_of_devices, const int* device_id)
{
    pgf_list_gpu* list = (pgf_list_gpu*)malloc(sizeof(pgf_list_gpu) * number_of_workers);
    memset(list, 0, sizeof(pgf_list_gpu) * number_of_workers);
    int dev = 0;
    for (int i = 0; i < number_of_workers; i++) {
        list[i].number_of_devices            = num_of_devices;
        list[i].device_id                    = device_id[dev % num_of_devices];
        list[i].lmax                         = -1;
        list[i].batch_size                   = batch_size;
        list[i].list_length                  = 0;
        list[i].coef_dynamic_alloc_size_gpu_ = 0;
        list[i].lmax_cpu_                    = (int*)malloc(sizeof(int) * list->batch_size);
        list[i].coef_offset_cpu_             = (int*)malloc(sizeof(int) * list->batch_size);
        list[i].rp_cpu_                      = (double3*)malloc(sizeof(double3) * list->batch_size);
        list[i].radius_cpu_                  = (double*)malloc(sizeof(double) * list->batch_size);
        list[i].zeta_cpu_                    = (double*)malloc(sizeof(double) * list->batch_size);
        list[i].coef_cpu_                    = (double*)malloc(sizeof(double) * list->batch_size * 8 * 8 * 8);
        list[i].coef_alloc_size_gpu_         = list->batch_size * 8 * 8 * 8;
        cudaSetDevice(list[i].device_id);
        cudaMalloc((void**)&list[i].radius_gpu_, sizeof(double) * list->batch_size);
        cudaMalloc((void**)&list[i].coef_offset_gpu_, sizeof(int) * list->batch_size);
        cudaMalloc((void**)&list[i].lmax_gpu_, sizeof(int) * list->batch_size);
        cudaMalloc((void**)&list[i].rp_gpu_, sizeof(double3) * list->batch_size);
        cudaMalloc((void**)&list[i].zeta_gpu_, sizeof(double) * list->batch_size);
        cudaMalloc((void**)&list[i].coef_gpu_, sizeof(double) * list->coef_alloc_size_gpu_);
        cudaStreamCreate(&list[i].stream);
        cublasCreate(&list[i].blas_handle);
        cublasSetStream(list[i].blas_handle, list[i].stream);
        cudaEventCreate(&list[i].event);
        list[i].next    = list + i + 1;
        list[i].running = false;
        dev++;
    }

    list[number_of_workers - 1].next = NULL;
    return list;
}

void
destroy_worker_list(pgf_list_gpu* const list)
{
    if (list == NULL)
        return;
    for (pgf_list_gpu* lst = list; lst->next; lst++) {
        cudaSetDevice(list->device_id);
        cudaFree(lst->coef_offset_gpu_);
        cudaFree(lst->lmax_gpu_);
        cudaFree(lst->rp_gpu_);
        cudaFree(lst->radius_gpu_);
        cudaFree(lst->zeta_gpu_);
        cudaFree(lst->coef_gpu_);
        cudaStreamDestroy(lst->stream);
        cublasDestroy(lst->blas_handle);
        cudaEventDestroy(lst->event);
        free(lst->lmax_cpu_);
        free(lst->coef_offset_cpu_);
        free(lst->rp_cpu_);
        free(lst->radius_cpu_);
        free(lst->zeta_cpu_);
        free(lst->coef_cpu_);
    }
    free(list);
}

inline void
release_gpu_resources(collocation_integration* handler)
{
    destroy_worker_list(handler->worker_list);
}

void
initialize_worker_list_on_gpu(collocation_integration* handle, const int number_of_devices, const int* device_id,
                              const int number_of_gaussian, const int number_of_workers)
{
    assert(handle != NULL);
    handle->worker_list_size   = number_of_workers;
    handle->number_of_gaussian = number_of_gaussian;

    handle->lmax = -1;

    /* we can inclrease this afterwards */
    /* right now only one list */

    handle->worker_list = create_worker_list(number_of_workers, number_of_gaussian, number_of_devices, device_id);
    initialize_grid_parameters_on_gpu(handle);
}

//******************************************************************************
// \brief Collocate a range of tasks which are destined for the same grid level.
// \author Ole Schuett
//******************************************************************************
void
collocate_one_grid_level_gpu(
    struct collocation_integration_* handler, const intl_task_list_t* task_list, const int first_task,
    const int last_task, const int func,
    /* const int grid_full_size[3], /\* size of the full grid *\/ */
    /* const int grid_local_size[3], /\* size of the local grid block *\/ */
    /* const int shift_local[3], /\* coordinates of the lower coordinates of the local grid window *\/ */
    /* const int border_width[3], /\* width of the borders *\/ */
    /* const double dh[3][3], /\* displacement vectors of the grid (cartesian) -> (ijk) *\/ */
    /* const double dh_inv[3][3], /\* (ijk) -> (x,y,z) *\/ */
    double* grid_)
{
    int lmin_diff[2], lmax_diff[2];
    grid_prepare_get_ldiffs_dgemm(func, lmin_diff, lmax_diff);

    int num_thread = 1;

#ifdef _OPENMP
#pragma omp parallel
    num_thread = omp_get_num_threads();
#endif

    initialize_worker_list_on_gpu(handler, 1, /* number of devices */
                                  handler->device_id, 4096, num_thread);

    // Using default(shared) because with GCC 9 the behavior around const changed:
    // https://www.gnu.org/software/gcc/gcc-9/porting_to.html
#pragma omp parallel default(shared)
    {

#ifdef _OPENMP
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif

        pgf_list_gpu* my_worker = &handler->worker_list[omp_get_thread_num()];
        reset_list_gpu(my_worker);
        my_worker->running = false;
        dgemm_params m1, m2;
        memset(&m1, 0, sizeof(dgemm_params));
        memset(&m2, 0, sizeof(dgemm_params));

        tensor work, subblock, pab, pab_prep, coef, alpha;

        // Allocate pab matrix for re-use across tasks.
        initialize_tensor_2(&pab, task_list->maxco, task_list->maxco);
        alloc_tensor(&pab);

        initialize_tensor_2(&work, task_list->maxco, task_list->maxco);
        alloc_tensor(&work);

        initialize_tensor_2(&subblock, task_list->maxco, task_list->maxco);
        alloc_tensor(&subblock);

        initialize_tensor_2(&pab_prep, task_list->maxco, task_list->maxco);
        alloc_tensor(&pab_prep);

        {
            const int lmax = task_list->lmax + max(lmax_diff[0], lmax_diff[1]);

            initialize_tensor_3(&coef, 2 * lmax + 1, 2 * lmax + 1, 2 * lmax + 1);
            alloc_tensor(&coef);

            initialize_tensor_4(&alpha, 3, lmax + 1, lmax + 1, 2 * lmax + 1);
            alloc_tensor(&alpha);
        }

        // Initialize variables to detect when a new subblock has to be fetched.
        int prev_block_num = -1, prev_iset = -1, prev_jset = -1;

#pragma omp for schedule(static)
        for (int itask = first_task; itask <= last_task; itask++) {
            // Define some convenient aliases.
            const intl_task_t* task        = &task_list->tasks[itask];
            const int iatom                = task->iatom - 1;
            const int jatom                = task->jatom - 1;
            const int iset                 = task->iset - 1;
            const int jset                 = task->jset - 1;
            const int ipgf                 = task->ipgf - 1;
            const int jpgf                 = task->jpgf - 1;
            const int ikind                = task_list->atom_kinds[iatom] - 1;
            const int jkind                = task_list->atom_kinds[jatom] - 1;
            const intl_basis_set_t* ibasis = task_list->basis_sets[ikind];
            const intl_basis_set_t* jbasis = task_list->basis_sets[jkind];
            const int ncoseta              = ncoset[ibasis->lmax[iset]];
            const int ncosetb              = ncoset[jbasis->lmax[jset]];
            const int ncoa                 = ibasis->npgf[iset] * ncoseta; // size of carthesian set
            const int ncob                 = jbasis->npgf[jset] * ncosetb;
            const int block_num            = task->block_num - 1;

            // Load subblock from buffer and decontract into Cartesian sublock pab.
            // The previous pab can be reused when only ipgf or jpgf has changed.
            if (block_num != prev_block_num || iset != prev_iset || jset != prev_jset) {
                prev_block_num = block_num;
                prev_iset      = iset;
                prev_jset      = jset;

                // Define some more convenient aliases.
                const int nsgf_seta = ibasis->nsgf_set[iset]; // size of spherical set
                const int nsgf_setb = jbasis->nsgf_set[jset];
                const int nsgfa     = ibasis->nsgf; // size of entire spherical basis
                const int nsgfb     = jbasis->nsgf;
                const int sgfa      = ibasis->first_sgf[iset] - 1; // start of spherical set
                const int sgfb      = jbasis->first_sgf[jset] - 1;
                const int maxcoa    = ibasis->maxco;
                const int maxcob    = jbasis->maxco;

                // Locate current matrix block within the buffer.
                const int block_offset = task_list->block_offsets[block_num]; // zero based
                double* const block    = &task_list->blocks_buffer[block_offset];

                initialize_tensor_2(&subblock, nsgf_setb, nsgf_seta);
                realloc_tensor(&subblock);

                initialize_tensor_2(&work, nsgf_setb, ncoa);
                realloc_tensor(&work);

                initialize_tensor_2(&pab, ncob, ncoa);
                realloc_tensor(&pab);

                if (iatom <= jatom) {
                    m1.op1   = 'N';
                    m1.op2   = 'N';
                    m1.m     = work.size[0];
                    m1.n     = work.size[1];
                    m1.k     = nsgf_seta;
                    m1.alpha = 1.0;
                    m1.beta  = 0.0;
                    m1.a     = block + sgfb * nsgfa + sgfa;
                    m1.lda   = nsgfa;
                    m1.b     = &ibasis->sphi[sgfa * maxcoa];
                    m1.ldb   = maxcoa;
                    m1.c     = work.data;
                    m1.ldc   = work.ld_;
                } else {
                    m1.op1   = 'T';
                    m1.op2   = 'N';
                    m1.m     = work.size[0];
                    m1.n     = work.size[1];
                    m1.k     = nsgf_seta;
                    m1.alpha = 1.0;
                    m1.beta  = 0.0;
                    m1.a     = block + sgfa * nsgfb + sgfb;
                    m1.lda   = nsgfb;
                    m1.b     = &ibasis->sphi[sgfa * maxcoa];
                    m1.ldb   = maxcoa;
                    m1.c     = work.data;
                    m1.ldc   = work.ld_;
                }

                dgemm_simplified(&m1, false);

                m2.op1   = 'T';
                m2.op2   = 'N';
                m2.m     = pab.size[0];
                m2.n     = pab.size[1];
                m2.k     = work.size[0];
                m2.alpha = 1.0;
                m2.beta  = 0.0;
                m2.a     = &jbasis->sphi[sgfb * maxcob];
                m2.lda   = maxcob;
                m2.b     = work.data;
                m2.ldb   = work.ld_;
                m2.c     = pab.data;
                m2.ldc   = pab.ld_;

                dgemm_simplified(&m2, false);

            } // end of block loading

            const double zeta[2] = {ibasis->zet[iset * ibasis->maxpgf + ipgf],
                                    jbasis->zet[jset * jbasis->maxpgf + jpgf]};

            const double* ra = &task_list->atom_positions[3 * iatom];
            int offset[2]    = {ipgf * ncoseta, jpgf * ncosetb};

            int lmax[2] = {ibasis->lmax[iset], jbasis->lmax[jset]};
            int lmin[2] = {ibasis->lmin[iset], jbasis->lmin[jset]};

            const double zetp = zeta[0] + zeta[1];
            const double f    = zeta[1] / zetp;
            const double rab2 = task->rab[0] * task->rab[0] + task->rab[1] * task->rab[1] + task->rab[2] * task->rab[2];
            const double prefactor = ((iatom == jatom) ? 1.0 : 2.0) * exp(-zeta[0] * f * rab2);

            double rp[3], rb[3];
            for (int i = 0; i < 3; i++) {
                rp[i] = ra[i] + f * task->rab[i];
                rb[i] = ra[i] + task->rab[i];
            }

            int lmin_prep[2];
            int lmax_prep[2];

            lmin_prep[0] = max(lmin[0] + lmin_diff[0], 0);
            lmin_prep[1] = max(lmin[1] + lmin_diff[1], 0);

            lmax_prep[0] = lmax[0] + lmax_diff[0];
            lmax_prep[1] = lmax[1] + lmax_diff[1];

            const int n1_prep = ncoset[lmax_prep[0]];
            const int n2_prep = ncoset[lmax_prep[1]];

            /* we do not reallocate memory. We initialized the structure with the maximum lmax of the all list already.
             */
            initialize_tensor_2(&pab_prep, n2_prep, n1_prep);
            realloc_tensor(&pab_prep);
            memset(pab_prep.data, 0, pab_prep.alloc_size_ * sizeof(double));

            /* grid_prepare_pab(func, */
            /*                  offset[0], offset[1], */
            /*                  lmax[0], lmin[0], */
            /*                  lmax[1], lmin[1], */
            /*                  zeta[0], */
            /*                  zeta[1], */
            /*                  pab.size[1], pab.size[0], (double (*)[pab.size[1]])pab.data, */
            /*                  n1_prep, n2_prep, (double (*)[n1_prep])pab_prep.data); */

            grid_prepare_pab_dgemm(func, offset, lmax, lmin, &zeta[0], &pab, &pab_prep);

            //   *** initialise the coefficient matrix, we transform the sum
            //
            // sum_{lxa,lya,lza,lxb,lyb,lzb} P_{lxa,lya,lza,lxb,lyb,lzb} *
            //         (x-a_x)**lxa (y-a_y)**lya (z-a_z)**lza (x-b_x)**lxb (y-a_y)**lya (z-a_z)**lza
            //
            // into
            //
            // sum_{lxp,lyp,lzp} P_{lxp,lyp,lzp} (x-p_x)**lxp (y-p_y)**lyp (z-p_z)**lzp
            //
            // where p is center of the product gaussian, and lp = la_max + lb_max
            // (current implementation is l**7)
            //

            /* precautionary tail since I probably intitialize data to NULL when I
             * initialize a new tensor. I want to keep the memory (I put a ridiculous
             * amount already) */

            initialize_tensor_4(&alpha, 3, lmax_prep[1] + 1, lmax_prep[0] + 1, lmax_prep[0] + lmax_prep[1] + 1);
            realloc_tensor(&alpha);

            const int lp = lmax_prep[0] + lmax_prep[1];

            initialize_tensor_3(&coef, lp + 1, lp + 1, lp + 1);
            realloc_tensor(&coef);

            // initialy cp2k stores coef_xyz as coef[z][y][x]. this is fine but I
            // need them to be stored as

            grid_prepare_alpha_dgemm(ra, rb, rp, lmax_prep, &alpha);

            //
            //   compute P_{lxp,lyp,lzp} given P_{lxa,lya,lza,lxb,lyb,lzb} and alpha(ls,lxa,lxb,1)
            //   use a three step procedure
            //   we don't store zeros, so counting is done using lxyz,lxy in order to have
            //   contiguous memory access in collocate_fast.F
            //

            // coef[x][z][y]
            grid_prepare_coef_dgemm(lmin_prep, lmax_prep, lp, prefactor, &alpha, &pab_prep, &coef);

            my_worker_is_running(my_worker);

            add_orbital_to_list(my_worker, coef.size[2] - 1, rp, task->radius, zetp, &coef);

            /* The list is full so we can start computation on GPU */
            if (my_worker->batch_size == my_worker->list_length) {
                handler->worker_list[thread_id].running = true;
                compute_collocation_gpu(my_worker);
            }
        }

        /* We may have a partial batch to do so we must complete them before doing anything else */
        if (handler->worker_list[thread_id].list_length) {
            /* ensure that the potential work running on the stream is done
             * before running the next batch */
            cudaSetDevice(my_worker->device_id);
            cudaStreamSynchronize(my_worker->stream);
            compute_collocation_gpu(my_worker);
        }

        // Merge thread local grids into shared grid.

        free(pab.data);
        free(pab_prep.data);
        free(work.data);
        free(subblock.data);
        free(coef.data);
        free(alpha.data);
    }

    cudaSetDevice(handler->worker_list->device_id);
    cudaStreamSynchronize(handler->worker_list->stream);

    double* tmp = NULL;

    if (handler->worker_list->number_of_devices != 1) {
        tmp = malloc(sizeof(double) * handler->grid.alloc_size_);
        memset(grid_, 0, sizeof(double) * handler->grid.alloc_size_);
    }

    for (int worker = 1; worker < handler->worker_list_size; worker++) {
        double alpha = 1.0;
        cudaSetDevice(handler->worker_list[worker].device_id);
        cudaStreamSynchronize(handler->worker_list[worker].stream);
        if (handler->worker_list->number_of_devices == 1) {
            cublasDaxpy(handler->worker_list->blas_handle, handler->grid.alloc_size_, &alpha,
                        handler->worker_list[worker].data_gpu_, 1, handler->worker_list[0].data_gpu_, 1);
        } else {
            if (handler->worker_list->device_id == handler->worker_list[worker].device_id) {
                cublasDaxpy(handler->worker_list->blas_handle, handler->grid.alloc_size_, &alpha,
                            handler->worker_list[worker].data_gpu_, 1, handler->worker_list[0].data_gpu_, 1);
            } else {
                cudaMemcpy(tmp, handler->worker_list[worker].data_gpu_, sizeof(double) * handler->grid.alloc_size_,
                           cudaMemcpyDeviceToHost);
                cblas_daxpy(handler->grid.alloc_size_, 1.0, tmp, 1, grid_, 1);
            }
        }
    }

    if (handler->worker_list->number_of_devices != 1) {
        cudaSetDevice(handler->worker_list->device_id);
        cudaMemcpy(tmp, handler->worker_list->data_gpu_, sizeof(double) * handler->grid.alloc_size_,
                   cudaMemcpyDeviceToHost);
        cblas_daxpy(handler->grid.alloc_size_, 1.0, tmp, 1, grid_, 1);
        free(tmp);
    } else {
        cudaStreamSynchronize(handler->worker_list->stream);
        cudaMemcpy(grid_, handler->worker_list->data_gpu_, sizeof(double) * handler->grid.alloc_size_,
                   cudaMemcpyDeviceToHost);
    }
}

void
grid_collocate_task_list_gpu(const int device_id, const intl_task_list_t* task_list, const bool orthorhombic, const int func,
                             const int nlevels, const int npts_global[nlevels][3], const int npts_local[nlevels][3],
                             const int shift_local[nlevels][3], const int border_width[nlevels][3],
                             const double dh[nlevels][3][3], const double dh_inv[nlevels][3][3], double* grid[nlevels])
{
    int first_task = 0;

    struct collocation_integration_* handler = collocate_create_handle();
    handler->device_id[0]                    = device_id;

    /* we can think about the multi gpu here */
    /*
      a device can be assigned to a worker so computation can be dispatched over
      multiple GPUs. Although it is already possible with the current
      implementation, it would still need to be adapted since we have GPU
      allocation when we create the workers.
     */

    /*
      second option : dispatch the different grid levels over the GPUs. In that
      case, it can be done by creating a handler for each gpu and indicate the
      gpu id
    */

    for (int level = 0; level < task_list->nlevels; level++) {
        const int last_task = first_task + task_list->tasks_per_level[level] - 1;

        initialize_basis_vectors(handler, dh[level], dh_inv[level]);
        verify_orthogonality(dh[level], handler->orthogonal);
        if (orthorhombic) {
            handler->orthogonal[0] = true;
            handler->orthogonal[1] = true;
            handler->orthogonal[2] = true;
        }
        initialize_tensor_3(&handler->grid, npts_local[level][2], npts_local[level][1], npts_local[level][0]);
        handler->grid.ld_ = npts_global[level][0];
        setup_global_grid_size(&handler->grid, &npts_global[level][0]);

        setup_grid_window(&handler->grid, shift_local[level], border_width[level],
                          task_list->tasks[first_task].border_mask);

        handler->grid.data = grid[level];

        collocate_one_grid_level_gpu(handler, task_list, first_task, last_task, func, grid[level]);

        first_task = last_task + 1;
    }

    collocate_destroy_handle(handler);
}
#endif
