#include <stdio.h>

#include <stdlib.h>
#include <assert.h>

#ifdef __USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "collocation_integration.h"

#if defined(__USE_GPU)

void add_orbital_to_list(pgf_list_gpu *const list,
                         const int lp,
                         const int cube_size[3],
                         const int cube_position[3],
                         const double roffset[3],
                         const double zetp,
                         const tensor *const coef)
{
    assert(list->batch_size > list->list_length);

    list->lmax_cpu_[list->list_length] = lp;
    list->cube_size_cpu_[3 * list->list_length] = cube_size[0];
    list->cube_size_cpu_[3 * list->list_length + 1] = cube_size[1];
    list->cube_size_cpu_[3 * list->list_length + 2] = cube_size[2];

    list->cube_position_cpu_[3 * list->list_length] = cube_position[0];
    list->cube_position_cpu_[3 * list->list_length + 1] = cube_position[1];
    list->cube_position_cpu_[3 * list->list_length + 2] = cube_position[2];

    list->roffset_cpu_[3 * list->list_length] = roffset[0];
    list->roffset_cpu_[3 * list->list_length + 1] = roffset[1];
    list->roffset_cpu_[3 * list->list_length + 2] = roffset[2];

    list->zeta_cpu_[list->list_length] = zetp;
    list->coef_offset_cpu_[0] = 0;
    if (list->list_length > 0) {
        list->coef_offset_cpu_[list->list_length] = list->coef_offset_cpu_[list->list_length - 1] + list->coef_previous_alloc_size_;
        list->coef_dynamic_alloc_size_gpu_ += coef->alloc_size_;
    } else {
        list->coef_dynamic_alloc_size_gpu_ = coef->alloc_size_;
    }
    list->coef_previous_alloc_size_ = coef->alloc_size_;

    if (list->coef_dynamic_alloc_size_gpu_ > list->coef_alloc_size_gpu_) {
        list->durty = true;
        list->coef_cpu_ = realloc(list->coef_cpu_, sizeof(double) * (list->coef_dynamic_alloc_size_gpu_ + 4096));
        list->coef_alloc_size_gpu_ = list->coef_dynamic_alloc_size_gpu_ + 4096;
    }

    if(list->lmax < lp)
        list->lmax = lp;
    memcpy(list->coef_cpu_ + list->coef_offset_cpu_[list->list_length], coef->data, sizeof(double) * coef->alloc_size_);
    list->list_length++;
}

void initialize_gpu_data(collocation_integration *handler)
{

    if (!handler->use_gpu)
        return;

    for (int i = 0; i < handler->worker_list_size; i++) {
        handler->worker_list[i].batch_size = handler->number_of_gaussian;
        handler->worker_list[i].list_length = 0;
        handler->worker_list[i].coef_dynamic_alloc_size_gpu_ = 0;
        handler->worker_list[i].cube_size_cpu_ = (int *)malloc(sizeof(int) * 3 * handler->worker_list[i].batch_size);
        handler->worker_list[i].lmax_cpu_ = (int *)malloc(sizeof(int) * handler->worker_list[i].batch_size);
        handler->worker_list[i].cube_position_cpu_ = (int *)malloc(sizeof(int) * 3 * handler->worker_list[i].batch_size);
        handler->worker_list[i].coef_offset_cpu_ = (int *)malloc(sizeof(int) * handler->worker_list[i].batch_size);
        handler->worker_list[i].roffset_cpu_ = (double *)malloc(sizeof(double) * 3 * handler->worker_list[i].batch_size);
        handler->worker_list[i].zeta_cpu_ = (double *)malloc(sizeof(double) * handler->worker_list[i].batch_size);
        handler->worker_list[i].coef_cpu_ = (double *)malloc(sizeof(double) * handler->worker_list[i].batch_size * 8 * 8 * 8);
        handler->worker_list[i].coef_alloc_size_gpu_ = handler->worker_list[i].batch_size * 8 * 8 * 8;

        cudaSetDevice(handler->gpu_id);

        cudaMalloc((void **)&handler->worker_list[i].cube_size_gpu_, sizeof(int) * 3 * handler->worker_list[i].batch_size);
        cudaMalloc((void **)&handler->worker_list[i].cube_position_gpu_, sizeof(int) * 3 * handler->worker_list[i].batch_size);
        cudaMalloc((void **)&handler->worker_list[i].coef_offset_gpu_, sizeof(int) * handler->worker_list[i].batch_size);
        cudaMalloc((void **)&handler->worker_list[i].lmax_gpu_, sizeof(int) * handler->worker_list[i].batch_size);

        cudaMalloc((void **)&handler->worker_list[i].roffset_gpu_, sizeof(double) * 3 * handler->worker_list[i].batch_size);
        cudaMalloc((void **)&handler->worker_list[i].zeta_gpu_, sizeof(double) * handler->worker_list[i].batch_size);
        cudaMalloc((void **)&handler->worker_list[i].coef_gpu_, sizeof(double) * handler->worker_list[i].coef_alloc_size_gpu_);
        cudaStreamCreate(&handler->worker_list[i].stream);
        cublasCreate(&handler->worker_list[i].blas_handle);
    }
}
void release_gpu_resources(collocation_integration *handler)
{
    if (!handler->use_gpu)
        return;

    for (int i = 0; i < handler->worker_list_size; i++) {
        cudaFree(handler->worker_list[i].cube_size_gpu_);
        cudaFree(handler->worker_list[i].cube_position_gpu_);
        cudaFree(handler->worker_list[i].coef_offset_gpu_);
        cudaFree(handler->worker_list[i].lmax_gpu_);
        cudaFree(handler->worker_list[i].roffset_gpu_);
        cudaFree(handler->worker_list[i].zeta_gpu_);
        cudaFree(handler->worker_list[i].coef_gpu_);
        cudaStreamDestroy(handler->worker_list[i].stream);
        cublasDestroy(handler->worker_list[i].blas_handle);

        free(handler->worker_list[i].cube_size_cpu_);
        free(handler->worker_list[i].lmax_cpu_);
        free(handler->worker_list[i].cube_position_cpu_);
        free(handler->worker_list[i].coef_offset_cpu_);
        free(handler->worker_list[i].roffset_cpu_);
        free(handler->worker_list[i].zeta_cpu_);
        free(handler->worker_list[i].coef_cpu_);
    }
    free(handler->worker_list);
}

void initialize_worker_list_on_gpu(collocation_integration *handle, const int device_id, const int number_of_gaussian, const bool use_gpu)
{
    assert(handle != NULL);
    handle->gpu_id = device_id;
    handle->use_gpu = use_gpu;
    handle->worker_list_size = 1;
    handle->number_of_gaussian = number_of_gaussian;

    if ((handle->gpu_id >= 0) && handle->use_gpu) {
        handle->lmax = -1;

        /* we can inclrease this afterwards */
        /* right now only one list */
        handle->worker_list = (pgf_list_gpu*) malloc(sizeof(pgf_list_gpu));
        memset(handle->worker_list, 0, sizeof(pgf_list_gpu));
        initialize_gpu_data(handle);
        handle->worker_list[0].list_length = 0;
        handle->worker_list[0].lmax = -1;
    }
}

#endif
