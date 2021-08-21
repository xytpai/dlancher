#include "../dlancher.h"
#include "devmem.h"


namespace dlancher {


void info() { GPU_info(); }
int count() { return _cudaDeviceGetCount(); }
int use(int device) { return _cudaDeviceSet(device); }
void reset() { _cudaDeviceReset(); }
int create_stream(size_t *stream) { return _cudaStreamStart((cudaStream_t*) stream); }
int async(size_t stream) { return _cudaStreamAsync((cudaStream_t)stream); }

namespace float32 {

float* malloc(int device, unsigned int len) 
    { return fmalloc(device, len); }
int memcpy(int dst_device, float *dst, int src_device, const float *src, unsigned int len) 
    { return fmemcpy(dst_device, dst, src_device, src, len); }
int memzeros(int device, float *ptr, unsigned int len) 
    { return fmemzeros(device, ptr, len); }
int memset(int device, float *ptr, float value, unsigned int len) 
    { return fmemset(device, ptr, value, len); }
void free(int device, float *ptr)
    { ffree(device, ptr); }
void print2d(int device, float *ptr, int height, int width)
    { fprint(device, ptr, height, width); }

int matmul(size_t stream, const float *a, bool transpose_a, int ah, int aw,
	const float *b, bool transpose_b, int bw,
    float *c, bool inc, float *bias_h=nullptr, float *bias_w=nullptr)
{
    return matmul_CU(a, (int)transpose_a, ah, aw, b, (int)transpose_b, bw, 
        c, (int)inc, bias_h, bias_w, (cudaStream_t)stream);
}

int add(size_t stream, float *output, const float *first, const float *second, const int len)
{
    return add_forward_CU(output, first, second, len, (cudaStream_t)stream);
}

}
}
