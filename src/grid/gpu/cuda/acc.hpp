#ifndef ACC_HPP
#define ACC_HPP
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK(status)                                                   \
    if (status != cudaSuccess) {                                        \
        fprintf(stderr, "ERROR: %s %s %d\n", cudaGetErrorString(status), __FILE__, \
                __LINE__);                                              \
        abort();                                                        \
    }

namespace acc {
    inline void *DeviceMalloc(const size_t size_) {
        void *ptr;
        CHECK(cudaMalloc(&ptr, size_));
        return ptr;
    }

    inline void DeviceFree(void *ptr) {
        if (ptr != nullptr)
            cudaFree(ptr);
    }

    inline void SetDevice(const int device_id_) {
        cudaSetDevice(device_id_);
    }

    inline void create_stream(cudaStream_t *stream) {
        cudaStreamCreate(stream);
    }

    inline void destroy_stream(cudaStream_t stream) {
        cudaStreamDestroy(stream);
    }

    inline void device_memset_async(void *ptr, int val, int size_, cudaStream_t stream_) {
        cudaMemsetAsync(ptr, 0, size_, stream_);
    }

    inline void CopyToDevice(void *dst, void *src, const size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }

    inline void CopyFromDevice(void *dst, void *src, const size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }

    inline void CopyToDeviceAsync(void *dst, void *src, const size_t size, cudaStream_t stream) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    }

    inline void CopyFromDeviceAsync(void *dst, void *src, const size_t size, cudaStream_t stream) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    }
};
#endif
