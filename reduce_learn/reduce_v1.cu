#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v1: 改进线程访问模式的归约实现
 * 改进点：从跨步模式改为连续线程参与模式
 * 优势：减少线程发散，提高线程利用率
 * 技术细节：前一半线程连续参与计算，避免warp内线程空闲
 * 优化方向：进一步优化内存访问模式和归约策略
 */

__global__ void reduce_v1(float *d_in, float *d_out, int n) { 
  __shared__ float s_data[THREAD_PER_BLOCK];
  float *d_in_begin = d_in + blockDim.x * blockIdx.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  
  // 数据加载（同v0）
  s_data[tid] = (id < n) ? d_in_begin[tid] : 0.0f;
  __syncthreads();

  // 改进的归约模式：连续线程参与
  // 从 0, 2, 4, 8, 16, 32 的跨步模式
  // 改为 0-31, 0-15, 0-7, 0-3, 0-1 的连续模式
  for (unsigned int s = 1; s < blockDim.x; s = s << 1) {
    // 前一半线程连续参与计算，提高利用率
    if (tid < blockDim.x / (2 * s)) {
      int index = tid * 2 * s;  // 计算要处理的元素索引
      s_data[index] += s_data[index + s];
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

  int block_num = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
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
    for (int j = 0; j < THREAD_PER_BLOCK; j++) {
      cur += a[i * THREAD_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 Grid(block_num, 1);
  dim3 Block(THREAD_PER_BLOCK, 1);
  reduce_v1<<<Grid, Block>>>(d_a, d_out, N);
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