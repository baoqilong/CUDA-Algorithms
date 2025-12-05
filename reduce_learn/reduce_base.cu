#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_baseline: 最基础的CUDA归约实现
 * 特点：直接在全局内存中进行归约操作
 * 缺点：全局内存访问速度慢，存在大量重复读写
 * 优点：实现简单，易于理解
 */

__global__ void reduce_baseline(float* d_in, float* d_out, int n) {
  // 计算当前block处理的输入数据起始位置
  // 每个block处理一段连续的数据
  float *d_inputbegin = d_in + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  
  // 在全局内存中进行归约
  // 使用跨步访问模式：1, 2, 4, 8, 16, ...
  for (int i = 1; i < blockDim.x; i <<= 1) {
    // 只有满足条件的线程参与计算
    // 这种模式会导致线程利用率逐渐降低
    if(tid % (2 * i) == 0) {
      d_inputbegin[tid] += d_inputbegin[tid + i];
    }
    __syncthreads();  // 同步block内的所有线程
  }
  
  // 将最终结果写回全局内存
  if(tid == 0)
    d_out[blockIdx.x] = d_inputbegin[tid];
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

  int block_size = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  float *out = (float *)calloc(block_size, sizeof(float));
  float *d_out;
  cudaMalloc((void **)&d_out, block_size * sizeof(float));
  float *res = (float *)malloc(block_size * sizeof(float));

  for (int i = 0; i < N; ++i) {
    a[i] = i % 100;
  }

  // CPU上的规约 - 作为参考结果
  for (int i = 0; i < block_size; ++i) {
    float cur = 0;
    for (int j = 0; j < THREAD_PER_BLOCK; j++) {
      cur += a[i * THREAD_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 Grid(block_size, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);
  reduce_baseline<<<Grid, Block>>>(d_a, d_out, N);
  cudaMemcpy(out, d_out, block_size * sizeof(float), cudaMemcpyDeviceToHost);
  if (check(out, res, block_size))
    printf("the ans is right\n");
  else {
    printf("the ans is wrong\n");
  }

  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
  free(res);
  return 0;
}