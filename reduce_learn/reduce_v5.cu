#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v6: 综合优化的归约实现
 * 特点：结合了前几个版本的所有优化技术
 * 优化策略：
 * 1. 多元素预处理减少线程数量
 * 2. 共享内存归约减少全局内存访问
 * 3. Warp级别优化最后阶段
 * 4. 模板化编译优化
 * 5. 手动循环展开避免分支判断
 * 性能：
 */

// 模板化的warp归约函数
// 根据blockSize在编译时选择不同的归约路径
template <unsigned int blockSize>
__device__ void warpreduce(volatile float *cache, int tid) {
  // 编译时条件判断，避免运行时开销
  if (blockSize >= 64)
    cache[tid] += cache[tid + 32];
  if (blockSize >= 32)
    cache[tid] += cache[tid + 16];
  if (blockSize >= 16)
    cache[tid] += cache[tid + 8];
  if (blockSize >= 8)
    cache[tid] += cache[tid + 4];
  if (blockSize >= 4)
    cache[tid] += cache[tid + 2];
  if (blockSize >= 2)
    cache[tid] += cache[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduce_v5(float *d_in, float *d_out, int n) {
  volatile __shared__ float s_data[THREAD_PER_BLOCK];
  float *d_in_begin = d_in + blockDim.x * blockIdx.x * 2;
  int tid = threadIdx.x;
  s_data[tid] = d_in_begin[tid] + d_in_begin[tid + blockDim.x];
  __syncthreads();

  // 编译时展开的归约循环
  // 根据blockSize在编译时选择不同的执行路径
  if (blockSize >= 512) {
    if (tid < 256) {
      s_data[tid] += s_data[tid + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128) {
      s_data[tid] += s_data[tid + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64) {
      s_data[tid] += s_data[tid + 64];
    }
    __syncthreads();
  }
  
  // 最后32个线程使用模板化的warp归约
  if (tid < 32)
    warpreduce<blockSize>(s_data, tid);

  if (tid == 0)
    d_out[blockIdx.x] = s_data[0];
}

template <typename T> bool check(T *out, T *res, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != res[i])
      return false;
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
  reduce_v5<THREAD_PER_BLOCK><<<Grid, Block>>>(d_a, d_out, N);
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