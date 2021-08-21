#pragma once

#include <iostream>
#include <vector>

namespace dlancher
{

void info();
int count();
int use(int device);
void reset();
int create_stream(size_t *stream);
int async(size_t stream);

namespace float32 {
float* malloc(int device, unsigned int len);
int memcpy(int dst_device, float *dst, int src_device, const float *src, unsigned int len);
int memzeros(int device, float *ptr, unsigned int len);
int memset(int device, float *ptr, float value, unsigned int len);
void free(int device, float *ptr);
void print2d(int device, float *ptr, int height, int width);

int matmul(size_t stream,
    const float *a, bool transpose_a, int ah, int aw,
	const float *b, bool transpose_b, int bw,
    float *c, bool inc, float *bias_h, float *bias_w);

int add(size_t stream, float *output, const float *first, const float *second, const int len);

}


} // namespace dlancher

