#ifndef UTILS_H_
#define UTILS_H_

inline void find_interval(const int start, const int end, const int *non_zero_elements_, int *zmin, int *zmax)
{
    int si;

    // loop over the table until we reach a 1
    for (si = start;(si < end - 1) && (non_zero_elements_[si] == 0); si++);

    // interval starts here
    *zmin = si;

    // now search where it ends;

    // loop over the table until we reach a 1
    for (;(si < (end - 1)) && (non_zero_elements_[si] == 1); si++);

    *zmax = si + non_zero_elements_[si];
}

inline int return_length_l(const int l) {
    static const int length_[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55};
    return length_[l];
}

inline int return_offset_l(const int l) {
    static const int offset_[10] = {0, 1, 4, 10, 20, 35, 56, 84, 120, 165};
    return offset_[l];
}

inline int return_linear_index_from_exponents(const int alpha, const int beta,
                                              const int gamma) {
    const int l = alpha + beta + gamma;
    return return_offset_l(l) + (l - alpha) * (l - alpha + 1) / 2 + gamma;
}

#endif
