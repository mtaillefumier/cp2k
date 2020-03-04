#ifndef TENSOR_LOCAL_H
#define TENSOR_LOCAL_H

#include <stdio.h>
#include <stdlib.h>

typedef struct tensor_ {
    int dim_;
    int size[4];
    size_t alloc_size_;
    size_t offsets[4];
    double *data;
    int ld_;
} tensor;

/* initialize a tensor structure for a tensor of dimension dim <= 4 */

inline void initialize_tensor(struct tensor_ *a, const int dim, const int *const sizes)
{
    a->dim_ = dim;
    for (int d = 0; d < dim; d++)
        a->size[d] = sizes[d];

    // we need proper alignment here. But can be done later
    a->ld_ = sizes[a->dim_ - 1];
    switch(a->dim_) {
    case 4: {
        a->offsets[0] = a->ld_ * a->size[1] * a->size[2];
        a->offsets[1] = a->ld_ * a->size[2];
        a->offsets[2] = a->ld_;
        break;
    }
    case 3: {
        a->offsets[0] = a->ld_ * a->size[1];
        a->offsets[1] = a->ld_;
    }
        break;
    case 2: { // matrix case
            a->offsets[0] = a->ld_;
        }
        break;
    case 1:
        break;
    }

    a->alloc_size_ = a->offsets[0] * a->size[0];
    return;
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void initialize_tensor_2(struct tensor_ *a, int n1, int n2)
{
    int size_[2] = {n1, n2};
    initialize_tensor(a, 2, size_);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void initialize_tensor_3(struct tensor_ *a, int n1, int n2, int n3)
{
    int size_[3] = {n1, n2, n3};
    initialize_tensor(a, 3, size_);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline void initialize_tensor_4(struct tensor_ *a, int n1, int n2, int n3, int n4)
{
    int size_[4] = {n1, n2, n3, n4};
    initialize_tensor(a, 4, size_);
}

/* initialize a tensor structure for a tensor of dimension dim = 2 */

inline tensor *create_tensor(const int dim, const int *sizes)
{
    tensor *a = (tensor *)malloc(sizeof(struct tensor_));
    initialize_tensor(a, dim, sizes);
    posix_memalign((void **)&a->data, 16, sizeof(double) * a->alloc_size_);
}

inline size_t tensor_return_memory_size(const struct tensor_ *const a)
{
    return a->alloc_size_;
}

inline void tensor_assign_memory(struct tensor_ *a, void *data)
{
    a->data = data;
}

inline int tensor_get_leading_dimension(struct tensor_ *a)
{
    return a->ld_;
}

inline int tensor_set_leading_dimension(struct tensor_ *a, const int ld)
{
    a->ld_ = ld;
}

inline void recompute_tensor_offsets(tensor *a)
{
    switch(a->dim_) {
    case 4: {
        a->offsets[0] = a->ld_ * a->size[1] * a->size[2];
        a->offsets[1] = a->ld_ * a->size[2];
        a->offsets[2] = a->ld_;
        break;
    }
    case 3: {
        a->offsets[0] = a->ld_ * a->size[1];
        a->offsets[1] = a->ld_;
    }
        break;
    case 2: { // matrix case
        a->offsets[0] = a->ld_;
    }
        break;
    case 1:
        break;
    }

}

#define idx4(a, i, j, k, l) a.data[ (i) * a.offsets[0] + (j) * a.offsets[1] + (k) * a.offsets[2] + (l)]
#define idx3(a, i, j, k) a.data[(i) * a.offsets[0] + (j) * a.offsets[1] + (k)]
#define idx2(a, i, j) a.data[(i) * a.offsets[0] + (j)]

#endif
