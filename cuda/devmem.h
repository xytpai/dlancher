#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#ifdef CUDNN_EN
#include <cudnn.h>
#endif


#define BLOCK_SIZE   32 
#define BLOCK_SIZE_X 512 


int _cudaDeviceGetCount();
int _cudaDeviceSet(int device);
int _cudaDeviceReset();
int _cudaStreamStart(cudaStream_t *pstream);
int _cudaStreamAsync(cudaStream_t stream);


#ifdef CUDNN_EN
int _cudnnCreate(cudnnHandle_t *phandle);
int _cudnnDestroy(cudnnHandle_t handle);
int _cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream);
#endif


void GPU_info();
float* fmalloc(int device, unsigned int len);
int fmemcpy(int dst_device, float *dst, int src_device, const float *src, unsigned int len);
int fmemzeros(int device, float *ptr, unsigned int len);
int fmemset(int device, float *ptr, float value, unsigned int len);
void ffree(int device, float *ptr);
void fprint(int device, float *ptr, int height, int width);

int matmul_CU(const float *A, int AMEM_T, int Ah, int Aw,
	const float *B, int BMEM_T, int Bw,
	float *C, int inc, float *bias_h, float *bias_w,
	cudaStream_t stream);

int add_forward_CU(float *output, const float *first, const float *second, const int len, cudaStream_t stream);
