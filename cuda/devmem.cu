#include "devmem.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define GPU_EN
#ifdef GPU_EN
#define GPU_D2H cudaMemcpyDeviceToHost
#define GPU_H2D cudaMemcpyHostToDevice
#define GPU_D2D cudaMemcpyDeviceToDevice
#endif


// PERR(func(),"ERR:func")
int PERR(int funcret, const char *errinfo)
{
	if (funcret) { 
		printf("PERR(%d):%s\n",funcret,errinfo);
		return funcret; 
	}
	return 0;
}


int _cudaDeviceGetCount()
{
#ifdef GPU_EN
	int num=0;
	if(PERR(cudaGetDeviceCount(&num),"ERR:cudaGetDeviceCount")) return -1;
	return num;
#else
	return 0;
#endif
}


// choose device
//_cudaDeviceReset(), _cudaStreamStart(), _cudaStreamAsync(), 
//_cudnnCreate(), _cudnnDestroy(), _cudnnSetStream()
int _cudaDeviceSet(int device)
{
#ifdef GPU_EN
	if(PERR(cudaSetDevice(device),"ERR:cudaSetDevice")) return -1;
	return 0;
#else
	return -1;
#endif
}


// free all resources of gpu
int _cudaDeviceReset()
{
#ifdef GPU_EN
	if(PERR(cudaDeviceReset(),"ERR:cudaDeviceReset")) return -1;
	return 0;
#else
	return -1;
#endif
}


int _cudaStreamStart(cudaStream_t *pstream)
{
#ifdef GPU_EN
	if(PERR(cudaStreamCreate(pstream),"ERR:cudaStreamCreate"))
		return -1;
	return 0;
#else
	return -1;
#endif
}


int _cudaStreamAsync(cudaStream_t stream)
{
#ifdef GPU_EN
	if(PERR(cudaStreamSynchronize(stream),"ERR:cudaStreamSynchronize"))
		return -1;
	if(PERR(cudaStreamDestroy(stream),"ERR:cudaStreamDestroy"))
		return -1;
	return 0;
#else
	return -1;
#endif
}


#ifdef CUDNN_EM
int _cudnnCreate(cudnnHandle_t *phandle)
{
#ifdef GPU_EN
	if(PERR(cudnnCreate(phandle),"ERR:cudnnCreate"))
		return -1;
	return 0;
#else
	return -1;
#endif
}


int _cudnnDestroy(cudnnHandle_t handle)
{
#ifdef GPU_EN
	if(PERR(cudnnDestroy(handle),"ERR:cudnnDestroy"))
		return -1;
	return 0;
#else
	return -1;
#endif
}


int _cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream)
{
#ifdef GPU_EN
	if(PERR(cudnnSetStream(handle,stream),"ERR:cudnnSetStream"))
		return -1;
	return 0;
#else
	return -1;
#endif
}
#endif


void GPU_info()
{
#ifdef GPU_EN
	int count;
	cudaGetDeviceCount(&count);
	for (int i=0; i<count; i++) {
		printf("device:%d\n", i);
		cudaSetDevice(i);
		int dev;
		cudaDeviceProp prop;
		if(PERR(cudaGetDevice(&dev),"ERR:cudaGetDevice")) return;
		if(PERR(cudaGetDeviceProperties(&prop, dev),"ERR:cudaGetDeviceProperties")) return;
		printf("device name:%s\n", prop.name);
		printf("compute ability:%d.%d\n", prop.major, prop.minor);
		printf("maxGridSize:%d,%d,%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("maxThreadPerBlock:%d\n", prop.maxThreadsPerBlock);
		printf("maxThreadDim:%d,%d,%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("multiProcessorCount:%d\n", prop.multiProcessorCount);
		printf("resPerBlock:%d(K)\n", prop.regsPerBlock / 1024);
		printf("sharedMemoryPerBolck:%ld(K)\n", prop.sharedMemPerBlock / 1024);
		printf("totleGlobalMemory:%ld(M)\n", prop.totalGlobalMem / (1024 * 1024));
		printf("warpSize:%d\n", prop.warpSize);
		printf("constanMemory:%ld(K)\n", prop.totalConstMem / 1024);
		if (i!=count-1) printf("\n");
	}
#endif
}


// cpu:-1, gpu:0~
float* fmalloc(int device, unsigned int len)
{
	float *ptr = NULL;

	if (device < 0) { //cpu
		ptr = (float*)malloc(len * sizeof(float));
		if (ptr == NULL) return NULL;
	}

#ifdef GPU_EN
	if (device >= 0) { //gpu
		if(PERR(cudaSetDevice(device),"ERR:cudaSetDevice")) return NULL; //set dev
		if(PERR(cudaMalloc((void**)&ptr, len * sizeof(float)),"ERR:cudaMalloc")) {
			cudaFree((void*)ptr); return NULL;
		}
	}
#endif

	return ptr;
}


int fmemcpy(int dst_device, float *dst, int src_device, const float *src, unsigned int len)
{
	unsigned int i;
	int ret = -1;

	if (dst == NULL || src == NULL) return -1;

	if (dst_device < 0 && src_device < 0) {//cpu<-cpu
		for (i = 0; i < len; i++) dst[i] = src[i];
		ret = 0;
	}

#ifdef GPU_EN
	if (dst_device >= 0 || src_device >= 0) { //gpu

		//memcpy
		if (dst_device < 0 && src_device >= 0) { //cpu<-gpu

			cudaStream_t stream;
			if(PERR(cudaSetDevice(src_device),"ERR:cudaSetDevice"))
				return -1;
			if(PERR(cudaStreamCreate(&stream),"ERR:cudaStreamCreate"))
				return -1;
			if(PERR(cudaMemcpyAsync(dst, src, len * sizeof(float), GPU_D2H, stream),"ERR:cudaMemcpyAsync"))
				return -1;
			if(PERR(cudaStreamSynchronize(stream),"ERR:cudaStreamSynchronize"))
				return -1;
			if(PERR(cudaStreamDestroy(stream),"ERR:cudaStreamDestroy"))
				return -1;

		}
		else if (dst_device >= 0 && src_device < 0) { //gpu<-cpu

			cudaStream_t stream;
			if(PERR(cudaSetDevice(dst_device),"ERR:cudaSetDevice"))
				return -1;
			if(PERR(cudaStreamCreate(&stream),"ERR:cudaStreamCreate"))
				return -1;
			if(PERR(cudaMemcpyAsync(dst, src, len * sizeof(float), GPU_H2D, stream),"ERR:cudaMemcpyAsync"))
				return -1;
			if(PERR(cudaStreamSynchronize(stream),"ERR:cudaStreamSynchronize"))
				return -1;
			if(PERR(cudaStreamDestroy(stream),"ERR:cudaStreamDestroy"))
				return -1;

		}
		else { //gpu<-gpu
			
			if (dst_device == src_device) { //same device
				
				cudaStream_t stream;
				if(PERR(cudaSetDevice(src_device),"ERR:cudaSetDevice"))
					return -1;
				if(PERR(cudaStreamCreate(&stream),"ERR:cudaStreamCreate"))
					return -1;
				if(PERR(cudaMemcpyAsync(dst, src, len * sizeof(float), GPU_D2D, stream),"ERR:cudaMemcpyAsync"))
					return -1;
				if(PERR(cudaStreamSynchronize(stream),"ERR:cudaStreamSynchronize"))
					return -1;
				if(PERR(cudaStreamDestroy(stream),"ERR:cudaStreamDestroy"))
					return -1;

			}
			else { // gpu_b<-gpu_a
				
				float *data_buffer = NULL;
				cudaStream_t stream_d2h,stream_h2d;

				data_buffer = (float*)malloc(len * sizeof(float));
				if (data_buffer == NULL) return -1;
				
				//TO HOST
				if(PERR(cudaSetDevice(src_device),"ERR:cudaSetDevice")) {
					free(data_buffer); return -1; 
				}
				if(PERR(cudaStreamCreate(&stream_d2h),"ERR:cudaStreamCreate")) {
					free(data_buffer); return -1;
				}
				if(PERR(cudaMemcpyAsync(data_buffer, src, len * sizeof(float), GPU_D2H, stream_d2h),"ERR:cudaMemcpyAsync")) {
					free(data_buffer); return -1; 
				}
				if(PERR(cudaStreamSynchronize(stream_d2h),"ERR:cudaStreamSynchronize")) {
					free(data_buffer); return -1;
				}
				if(PERR(cudaStreamDestroy(stream_d2h),"ERR:cudaStreamDestroy")) {
					free(data_buffer); return -1;
				}

				//TO DEVICE
				if(PERR(cudaSetDevice(dst_device),"ERR:cudaSetDevice")) {
					free(data_buffer); return -1; 
				}
				if(PERR(cudaStreamCreate(&stream_h2d),"ERR:cudaStreamCreate")) {
					free(data_buffer); return -1;
				}
				if(PERR(cudaMemcpyAsync(dst, data_buffer, len * sizeof(float), GPU_H2D, stream_h2d),"ERR:cudaMemcpyAsync")) {
					free(data_buffer); return -1; 
				}
				if(PERR(cudaStreamSynchronize(stream_h2d),"ERR:cudaStreamSynchronize")) {
					free(data_buffer); return -1;
				}
				if(PERR(cudaStreamDestroy(stream_h2d),"ERR:cudaStreamDestroy")) {
					free(data_buffer); return -1;
				}
				
				free(data_buffer);
			}
		}
		ret = 0;
	}
#endif

	return ret;
}


int fmemzeros(int device, float *ptr, unsigned int len)
{
	unsigned int i;
	int ret = -1;

	if (ptr == NULL) return -1;

	if (device < 0) {
		for (i = 0; i < len; i++) ptr[i] = 0;
		ret = 0;
	}

#ifdef GPU_EN
	if (device >= 0) {
		cudaStream_t stream;
		if(PERR(cudaSetDevice(device),"ERR:cudaSetDevice"))
			return -1;
		if(PERR(cudaStreamCreate(&stream),"ERR:cudaStreamCreate"))
			return -1;
		if(PERR(cudaMemsetAsync(ptr, 0, sizeof(float)*len, stream),"ERR:cudaMemsetAsync"))
			return -1;
		if(PERR(cudaStreamSynchronize(stream),"ERR:cudaStreamSynchronize"))
			return -1;
		if(PERR(cudaStreamDestroy(stream),"ERR:cudaStreamDestroy"))
			return -1;
		ret = 0;
	}
#endif

	return ret;
}


#ifdef GPU_EN
__global__ void fmemset_Kernel(float *ptr, float value, unsigned int len)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	unsigned int x = bx*BLOCK_SIZE_X + tx;
	if (x < len) {
		ptr[x] = value;
	}
}
#endif


int fmemset(int device, float *ptr, float value, unsigned int len)
{
	unsigned int i;
	int ret = -1;

	if (ptr == NULL) return -1;

	if (device < 0) {
		for (i = 0; i < len; i++) ptr[i] = value;
		ret = 0;
	}

#ifdef GPU_EN
	if (device >= 0) {

		cudaStream_t stream;
		if(PERR(cudaSetDevice(device),"ERR:cudaSetDevice"))
			return -1;
		if(PERR(cudaStreamCreate(&stream),"ERR:cudaStreamCreate"))
			return -1;

		dim3 grid(len / BLOCK_SIZE_X + 1), block(BLOCK_SIZE_X);
		fmemset_Kernel<<<grid, block, 0, stream>>>(ptr, value, len);

		if(PERR(cudaStreamSynchronize(stream),"ERR:cudaStreamSynchronize"))
			return -1;
		if(PERR(cudaStreamDestroy(stream),"ERR:cudaStreamDestroy"))
			return -1;
		
		ret = 0;
	}
#endif

	return ret;
}


void ffree(int device, float *ptr)
{
	if (ptr == NULL) return;

	if (device < 0) {
		free(ptr);
	}

#ifdef GPU_EN
	if (device >= 0) {
		if(PERR(cudaSetDevice(device),"ERR:cudaSetDevice"))
			return;
		if(PERR(cudaFree(ptr),"ERR:cudaFree"))
			return;
	}
#endif

	ptr = NULL;
}


void fprint(int device, float *ptr, int height, int width)
{
	int h, w, ret;
	float *fcpu = fmalloc(-1, height*width);
	ret = fmemcpy(-1, fcpu, device, ptr, height*width);
	if(ret) {
		ffree(-1, fcpu);
		return;
	}
	for (h = 0; h < height; h++) {
		for (w = 0; w < width; w++) {
			printf("%10.4f", fcpu[h*width + w]);
		}
		printf("\n");
	}
	printf("\n");
	ffree(-1, fcpu);
}
