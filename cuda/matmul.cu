#include "devmem.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void matmul_Kernel(const float *A, int AMEM_T, int Ah, int Aw,
	const float *B, int BMEM_T, int Bw,
	float *C, int inc, float *bias_h, float *bias_w)
{
	int a, b, k;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int x = bx * BLOCK_SIZE + tx;
	int y = by * BLOCK_SIZE + ty;
	int ct = 0;
	int aBegin, aEnd, aStep, bBegin, bStep;

	if (!AMEM_T) {
		aBegin = Aw * (by*BLOCK_SIZE);//A(0,blockbegin_y)
		aStep = BLOCK_SIZE;//offsetA
		aEnd = aBegin + Aw - 1;
		
	}
	else {
		aBegin = (by*BLOCK_SIZE);
		aStep = Ah * BLOCK_SIZE;
		aEnd = aBegin + (Aw - 1)*Ah;
	}

	if (!BMEM_T) {
		bBegin = (BLOCK_SIZE*bx);//B(bx,0)
		bStep = BLOCK_SIZE * Bw;//offsetB
	}
	else {
		bBegin = (BLOCK_SIZE*bx)*Aw;
		bStep = BLOCK_SIZE;
	}

	float cSub = 0;
	for (a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
	{
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		if (!AMEM_T) {
			if (y < Ah&&ct + tx < Aw)
				As[ty][tx] = A[a + Aw * ty + tx];
			else
				As[ty][tx] = 0;
		}
		else {
			if (y < Ah&&ct + tx < Aw)
				As[ty][tx] = A[a + Ah * tx + ty];
			else
				As[ty][tx] = 0;
		}

		if (!BMEM_T) {
			if (ct + ty < Aw&&x < Bw)
				Bs[ty][tx] = B[b + Bw * ty + tx];
			else
				Bs[ty][tx] = 0;
		}
		else {
			if (ct + ty < Aw&&x < Bw)
				Bs[ty][tx] = B[b + Aw * tx + ty];
			else
				Bs[ty][tx] = 0;
		}

		__syncthreads();

		for (k = 0; k < BLOCK_SIZE; ++k)
		{
			cSub += As[ty][k] * Bs[k][tx];
		}
		ct += BLOCK_SIZE;

		__syncthreads();
	}

	if (y < Ah&&x < Bw) {

		if (bias_h) cSub += bias_h[y];
		if (bias_w) cSub += bias_w[x];

		if (!inc) C[y*Bw + x] = cSub;
		else C[y*Bw + x] += cSub;
	}
}


int matmul_CU(const float *A, int AMEM_T, int Ah, int Aw,
	const float *B, int BMEM_T, int Bw,
	float *C, int inc, float *bias_h, float *bias_w,
	cudaStream_t stream)
{
	dim3 grid(Bw / BLOCK_SIZE + 1, Ah / BLOCK_SIZE + 1), block(BLOCK_SIZE, BLOCK_SIZE);
	matmul_Kernel << < grid, block, 0, (cudaStream_t)stream >> > (A, AMEM_T, Ah, Aw, B, BMEM_T, Bw, C, inc, bias_h, bias_w);
	return 0;
}


__global__ void add_forward_kernel(float *output, const float *first, const float *second, const int len)
{
    int i = blockIdx.x*512 + threadIdx.x;
    if (i >= len) return;
    output[i] = first[i] + second[i];
}


int add_forward_CU(float *output, const float *first, const float *second, const int len, cudaStream_t stream)
{
    dim3 grid(len/512+1), block(512);
    add_forward_kernel << < grid, block, 0, stream >> > (output, first, second, len);
    return 0;
}
