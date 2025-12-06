#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v3: 多元素预处理的归约实现
 * 改进点：每个线程处理多个元素进行预归约
 * 优势：
 * - 减少后续归约阶段的线程数量
 * - 提高并行度利用率
 * - 减少线程调度开销
 * 技术细节：每个线程负责两个元素的初始加载和预归约
 */

__global__ void reduce_v3(float *d_in, float *d_out, int n) { 
  __shared__ float s_data[THREAD_PER_BLOCK];
  
  // 每个线程处理两个元素，起始位置偏移乘以2
  float *d_in_begin = d_in + blockDim.x * blockIdx.x * 2;
  int tid = threadIdx.x;
  
  // 每个线程加载并预归约两个元素
  // 这样只需要一半的线程就能处理相同数量的数据
  s_data[tid] = 0;
  for (int i = 0; i < blockDim.x; i += blockDim.x) {
    s_data[tid] = d_in_begin[tid] + d_in_begin[tid + blockDim.x];
  }
  __syncthreads();

  // 后续的共享内存归约同v2
  for (unsigned int s = blockDim.x / 2; s != 0; s = s >> 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) d_out[blockIdx.x] = s_data[0];
}

template <typename T>
bool check(T *out, T *res, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != res[i]) return false;
  }
  return true;
}

int main() {
  const int N = 32 * 1024 * 1024 - 1;
  float *a = (float *)malloc(N * sizeof(float));
  float *d_a;
  cudaMalloc((void **)&d_a, N * sizeof(float));

  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK / 2;
  float *out = (float *)calloc(block_num, sizeof(float));
  float *d_out;
  cudaMalloc((void **)&d_out, block_num * sizeof(float));
  float *res = (float *)malloc(block_num * sizeof(float));

  for (int i = 0; i < N; ++i) {
    a[i] = i % 100;
  }

  // CPU上的规约
  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < THREAD_PER_BLOCK * 2; j++) {
      cur += a[i * THREAD_PER_BLOCK * 2 + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 Grid(block_num, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);
  reduce_v3<<<Grid, Block>>>(d_a, d_out, N);
  cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
  if (check(out, res, block_num))
    printf("the ans is right\n");
  else {
    printf("the ans is wrong\n");
    for (int i = 0; i < block_num; i++) {
      printf("%lf ", out[i]);
    }
    printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
  free(res);
  return 0;
}