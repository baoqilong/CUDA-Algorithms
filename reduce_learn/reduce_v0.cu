#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v0: 引入共享内存的归约实现
 * 改进点：使用共享内存替代全局内存进行归约操作
 * 优势：共享内存访问速度比全局内存快
 * 问题：
 * - 归约模式仍为低效的跨步模式
 * - 存在严重的线程发散和bank冲突
 * 优化方向：后续版本将改进归约模式和访问模式
 */

__global__ void reduce_v0(float *d_in, float *d_out, int n) { 
  // 声明共享内存数组
  __shared__ float s_data[THREAD_PER_BLOCK];
  
  // 计算当前block处理的数据起始位置
  float *d_in_begin = d_in + blockDim.x * blockIdx.x;
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int tid = threadIdx.x;
  
  // 将数据从全局内存加载到共享内存
  // 边界检查：防止数组越界
  s_data[tid] = (id < n) ? d_in_begin[tid] : 0.0f;
  __syncthreads();

  // 在共享内存中进行归约
  // 仍然使用跨步访问模式，但访问的是共享内存
  for (unsigned int s = 1; s < blockDim.x; s = s << 1) {
    if (tid % (2 * s) == 0) {
      s_data[tid] += s_data[tid + s];
    }
    __syncthreads();
  }
  
  // 将最终结果写回全局内存
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
  reduce_v0<<<Grid, Block>>>(d_a, d_out, N);
  cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
  if (check(out, res, block_num))
    printf("the ans is right\n");
  else {
    printf("the ans is wrong\n");
    // for (int i = 0; i < block_num; i++) {
    //   printf("%lf ", out[i]);
    // }
    // printf("\n");
  }

  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
  free(res);
  return 0;
}