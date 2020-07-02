#include <stdio.h>

#include <stdlib.h>
#include <assert.h>

#ifdef __COLLOCATE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "collocation_integration.h"

#if defined(__COLLOCATE_GPU)

pgf_list_gpu* create_worker_list(const int number_of_workers, const int batch_size, const int device_id);

void destroy_worker_list(pgf_list_gpu* const list);

void
add_orbital_to_list(pgf_list_gpu* const list, const int lp, const int cube_size[3], const int cube_position[3],
                    const double roffset[3], const double zetp, const tensor* const coef)
{
    assert(list->batch_size > list->list_length);

    list->lmax_cpu_[list->list_length]              = lp;
    list->cube_size_cpu_[3 * list->list_length]     = cube_size[0];
    list->cube_size_cpu_[3 * list->list_length + 1] = cube_size[1];
    list->cube_size_cpu_[3 * list->list_length + 2] = cube_size[2];


    list->cube_position_cpu_[3 * list->list_length]     = cube_position[0];
    list->cube_position_cpu_[3 * list->list_length + 1] = cube_position[1];
    list->cube_position_cpu_[3 * list->list_length + 2] = cube_position[2];

    list->roffset_cpu_[3 * list->list_length]     = roffset[0];
    list->roffset_cpu_[3 * list->list_length + 1] = roffset[1];
    list->roffset_cpu_[3 * list->list_length + 2] = roffset[2];

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

void
initialize_gpu_data(collocation_integration* handler)
{

    if (!handler->use_gpu)
        return;

    handler->worker_list =
        create_worker_list(handler->worker_list_size, handler->number_of_gaussian, handler->device_id);
}

pgf_list_gpu*
create_worker_list(const int number_of_workers, const int batch_size, const int device_id)
{
    pgf_list_gpu* list = (pgf_list_gpu*)malloc(sizeof(pgf_list_gpu) * number_of_workers);
    for (int i = 0; i < number_of_workers; i++) {
        list[i].device_id                    = device_id;
        list[i].lmax                         = -1;
        list[i].batch_size                   = batch_size;
        list[i].list_length                  = 0;
        list[i].coef_dynamic_alloc_size_gpu_ = 0;
        list[i].cube_size_cpu_               = (int*)malloc(sizeof(int) * 3 * list->batch_size);
        list[i].lmax_cpu_                    = (int*)malloc(sizeof(int) * list->batch_size);
        list[i].cube_position_cpu_           = (int*)malloc(sizeof(int) * 3 * list->batch_size);
        list[i].coef_offset_cpu_             = (int*)malloc(sizeof(int) * list->batch_size);
        list[i].roffset_cpu_                 = (double*)malloc(sizeof(double) * 3 * list->batch_size);
        list[i].zeta_cpu_                    = (double*)malloc(sizeof(double) * list->batch_size);
        list[i].coef_cpu_                    = (double*)malloc(sizeof(double) * list->batch_size * 8 * 8 * 8);
        list[i].coef_alloc_size_gpu_         = list->batch_size * 8 * 8 * 8;
        cudaSetDevice(list[i].device_id);

        cudaMalloc((void**)&list[i].cube_size_gpu_, sizeof(int) * 3 * list->batch_size);
        cudaMalloc((void**)&list[i].cube_position_gpu_, sizeof(int) * 3 * list->batch_size);
        cudaMalloc((void**)&list[i].coef_offset_gpu_, sizeof(int) * list->batch_size);
        cudaMalloc((void**)&list[i].lmax_gpu_, sizeof(int) * list->batch_size);

        cudaMalloc((void**)&list[i].roffset_gpu_, sizeof(double) * 3 * list->batch_size);
        cudaMalloc((void**)&list[i].zeta_gpu_, sizeof(double) * list->batch_size);
        cudaMalloc((void**)&list[i].coef_gpu_, sizeof(double) * list->coef_alloc_size_gpu_);
        cudaStreamCreate(&list[i].stream);
        cublasCreate(&list[i].blas_handle);
        list[i].next = list + i + 1;
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
        cudaFree(lst->cube_size_gpu_);
        cudaFree(lst->cube_position_gpu_);
        cudaFree(lst->coef_offset_gpu_);
        cudaFree(lst->lmax_gpu_);
        cudaFree(lst->roffset_gpu_);
        cudaFree(lst->zeta_gpu_);
        cudaFree(lst->coef_gpu_);
        cudaStreamDestroy(lst->stream);
        cublasDestroy(lst->blas_handle);

        free(lst->cube_size_cpu_);
        free(lst->lmax_cpu_);
        free(lst->cube_position_cpu_);
        free(lst->coef_offset_cpu_);
        free(lst->roffset_cpu_);
        free(lst->zeta_cpu_);
        free(lst->coef_cpu_);
    }
    free(list);
}

void
release_gpu_resources(collocation_integration* handler)
{
    if (!handler->use_gpu)
        return;
    destroy_worker_list(handler->worker_list);
}

void
initialize_worker_list_on_gpu(collocation_integration* handle,
                              const int device_id,
                              const int number_of_gaussian,
                              const int number_of_workers,
                              const bool use_gpu)
{
    if ((!use_gpu) || (device_id < 0))
        return;

    assert(handle != NULL);
    handle->device_id          = device_id;
    handle->use_gpu            = use_gpu;
    handle->worker_list_size   = number_of_workers;
    handle->number_of_gaussian = number_of_gaussian;

    if ((handle->device_id >= 0) && handle->use_gpu) {
        handle->lmax = -1;

        /* we can inclrease this afterwards */
        /* right now only one list */

        handle->worker_list = create_worker_list(number_of_workers, number_of_gaussian, device_id);
    }
}

#endif
