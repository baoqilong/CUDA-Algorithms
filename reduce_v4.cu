#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v4: Warp优化的归约实现
 * 改进点：最后32个线程（一个warp）使用无同步归约
 * 利用warp内线程执行SIMT的特性，避免__syncthreads()
 */

// Warp级别的归约函数
// 手动展开最后6步归约，避免循环开销
__device__ void warpreduce(volatile float* s_data, int tid) {
  s_data[tid] += s_data[tid + 32];  // 32->16
  s_data[tid] += s_data[tid + 16];  // 16->8
  s_data[tid] += s_data[tid + 8];   // 8->4
  s_data[tid] += s_data[tid + 4];   // 4->2
  s_data[tid] += s_data[tid + 2];   // 2->1
  s_data[tid] += s_data[tid + 1];   // 1->最终结果
}

__global__ void reduce_v4(float *d_in, float *d_out, int n) {
  volatile __shared__ float s_data[THREAD_PER_BLOCK];
  
  // 同v3：每个线程处理两个元素
  float *d_in_begin = d_in + blockDim.x * blockIdx.x * 2;
  int tid = threadIdx.x;
  s_data[tid] = d_in_begin[tid] + d_in_begin[tid + blockDim.x];
  __syncthreads();

  // 前一半线程的归约，同v3
  for (unsigned int s = blockDim.x / 2; s > 32; s = s >> 1) {
    if (tid < s) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }
  
  // 最后32个线程使用warp级无同步归约
  // 利用warp内线程同步执行的特性
  if (tid < 32) {
    warpreduce(s_data, tid);
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

  int block_size = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK / 2;
  float *out = (float *)calloc(block_size, sizeof(float));
  float *d_out;
  cudaMalloc((void **)&d_out, block_size * sizeof(float));
  float *res = (float *)malloc(block_size * sizeof(float));

  for (int i = 0; i < N; ++i) {
    a[i] = i % 100;
  }

  // CPU上的规约
  for (int i = 0; i < block_size; ++i) {
    float cur = 0;
    for (int j = 0; j < THREAD_PER_BLOCK * 2; j++) {
      cur += a[i * THREAD_PER_BLOCK * 2 + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 Grid(block_size, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);
  reduce_v4<<<Grid, Block>>>(d_a, d_out, N);
  cudaMemcpy(out, d_out, block_size * sizeof(float), cudaMemcpyDeviceToHost);
  if (check(out, res, block_size))
    printf("the ans is right\n");
  else {
    printf("the ans is wrong\n");
    for (int i = 0; i < block_size; i++) {
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