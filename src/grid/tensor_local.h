#ifndef TENSOR_LOCAL_H
#define TENSOR_LOCAL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <malloc.h>
#include <string.h>
#ifdef __COLLOCATE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

typedef struct tensor_
{
    int dim_;
    int size[4];
    struct tensor_* block;
    size_t alloc_size_;
    size_t old_alloc_size_;
    size_t offsets[4];
    int blockDim[3];
    double* data;
    unsigned int ld_;
    size_t unblocked_size[4];
    bool blocked_decomposition;
} tensor;

extern void initialize_tensor_blocked(struct tensor_* a, const int dim, const int* const sizes,
                                      const int* const blockDim);

/* initialize a tensor structure for a tensor of dimension dim <= 4 */

inline void
initialize_tensor(struct tensor_* a, const int dim, const int* const sizes)
{
    if (a == NULL)
        return;

    a->block = NULL;
    a->dim_  = dim;
    for (int d = 0; d < dim; d++)
        a->size[d] = sizes[d];

    // we need proper alignment here. But can be done later
    /* a->ld_ = (sizes[a->dim_ - 1] / 32 + 1) * 32; */
    a->ld_ = sizes[a->dim_ - 1];
    switch (a->dim_) {
        case 4: {
            a->offsets[0] = a->ld_ * a->size[1] * a->size[2];
            a->offsets[1] = a->ld_ * a->size[2];
            a->offsets[2] = a->ld_;
            break;
        }
        case 3: {
            a->offsets[0] = a->ld_ * a->size[1];
            a->offsets[1] = a->ld_;
        } break;
        case 2: { // matrix case
            a->offsets[0] = a->ld_;
        } break;
        case 1:
            break;
    }

    a->alloc_size_           = a->offsets[0] * a->size[0];
    a->blocked_decomposition = false;
    return;
}

/* inline void allocate_tensor_on_gpu(struct tensor_ *a) */
/* { */
/* #ifdef __USE_GPU */
/*     cudaMalloc((void**)&a->data_gpu_, sizeof(double) * a->alloc_size_); */
/* #else */
/*     printf("GPU support is off\n"); */
/*     a->data_gpu_ = NULL; */
/* #endif */
/* } */

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void
initialize_tensor_2(struct tensor_* a, int n1, int n2)
{
    if (a == NULL)
        return;

    int size_[2] = {n1, n2};
    initialize_tensor(a, 2, size_);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void
initialize_tensor_3(struct tensor_* a, int n1, int n2, int n3)
{
    if (a == NULL)
        return;
    int size_[3] = {n1, n2, n3};
    initialize_tensor(a, 3, size_);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void
initialize_tensor_4(struct tensor_* a, int n1, int n2, int n3, int n4)
{
    if (a == NULL)
        return;
    int size_[4] = {n1, n2, n3, n4};
    initialize_tensor(a, 4, size_);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void
initialize_tensor_blocked_2(struct tensor_* a, int n1, int n2, int* blockdim)
{
    if (a == NULL)
        return;

    int size_[2] = {n1, n2};

    initialize_tensor_blocked(a, 2, size_, blockdim);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void
initialize_tensor_blocked_3(struct tensor_* a, int n1, int n2, int n3, int* blockdim)
{
    if (a == NULL)
        return;
    int size_[3] = {n1, n2, n3};

    initialize_tensor_blocked(a, 3, size_, blockdim);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void
initialize_tensor_blocked_4(struct tensor_* a, int n1, int n2, int n3, int n4, int* blockdim)
{
    if (a == NULL)
        return;
    int size_[4] = {n1, n2, n3, n4};
    initialize_tensor_blocked(a, 4, size_, blockdim);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline tensor*
create_tensor(const int dim, const int* sizes)
{
    tensor* a = (tensor*)malloc(sizeof(struct tensor_));

    if (a == NULL)
        abort();

    initialize_tensor(a, dim, sizes);
    a->data = (double*)memalign(64, sizeof(double) * a->alloc_size_);
    if (a->data == NULL)
        abort();
    a->old_alloc_size_ = a->alloc_size_;
    return a;
}

/* destroy a tensor created with the function above */
inline void
destroy_tensor(tensor* a)
{
    if (a->block != NULL) {
        if (a->block->data)
            free(a->block->data);
        free(a->block);
    }

    if (a->data)
        free(a->data);
    free(a);
}

inline size_t
tensor_return_memory_size(const struct tensor_* const a)
{
    if (a == NULL)
        abort();

    return a->alloc_size_;
}

inline void
tensor_assign_memory(struct tensor_* a, void* data)
{
    if (a == NULL)
        abort();
    a->data = (double*)data;
}

inline int
tensor_get_leading_dimension(struct tensor_* a)
{
    if (a == NULL)
        abort();
    return a->ld_;
}

inline void
tensor_set_leading_dimension(struct tensor_* a, const int ld)
{
    if (a == NULL)
        abort();
    a->ld_ = ld;
}

inline void
recompute_tensor_offsets(struct tensor_* a)
{
    if (a == NULL)
        abort();

    switch (a->dim_) {
        case 5: {
            a->offsets[0] = a->ld_ * a->size[1] * a->size[2] * a->size[3];
            a->offsets[1] = a->ld_ * a->size[1] * a->size[2];
            a->offsets[2] = a->ld_ * a->size[2];
            a->offsets[3] = a->ld_;
            break;
        }
        case 4: {
            a->offsets[0] = a->ld_ * a->size[1] * a->size[2];
            a->offsets[1] = a->ld_ * a->size[2];
            a->offsets[2] = a->ld_;
            break;
        }
        case 3: {
            a->offsets[0] = a->ld_ * a->size[1];
            a->offsets[1] = a->ld_;
        } break;
        case 2: { // matrix case
            a->offsets[0] = a->ld_;
        } break;
        case 1:
            break;
    }
}

inline size_t
compute_memory_space_tensor_3(const int n1, const int n2, const int n3)
{
    return (n1 * n2 * n3);
}

inline size_t
compute_memory_space_tensor_4(const int n1, const int n2, const int n3, const int n4)
{
    return (n1 * n2 * n3 * n4);
}

extern size_t realloc_tensor(tensor* t);

#define idx5(a, i, j, k, l, m) a.data[(i)*a.offsets[0] + (j)*a.offsets[1] + (k)*a.offsets[2] + (l)*a.offsets[3] + m]
#define idx4(a, i, j, k, l) a.data[(i)*a.offsets[0] + (j)*a.offsets[1] + (k)*a.offsets[2] + (l)]
#define idx3(a, i, j, k) a.data[(i)*a.offsets[0] + (j)*a.offsets[1] + (k)]
#define idx2(a, i, j) a.data[(i)*a.offsets[0] + (j)]

#endif
