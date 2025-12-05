#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v6: 优化的CUDA归约算法实现
 * 特点：
 * 1. 每个线程处理多个元素，减少线程块数量
 * 2. 使用warp级别的原生指令优化最后32个线程的归约
 * 3. 完全展开循环以避免分支判断
 */

/**
 * @brief warp级别的归约函数
 * @tparam blockSize 线程块大小
 * @param cache 共享内存数组
 * @param tid 线程ID
 * 
 * 使用volatile关键字确保编译器不会优化掉内存访问
 * 手动展开最后6步归约（32->16->8->4->2->1）
 */
template <unsigned int blockSize>
__device__ void warpreduce(volatile float *cache, int tid) {
  // 当blockSize >= 64时，处理32个元素的归约
  if (blockSize >= 64)
    cache[tid] += cache[tid + 32];
  // 处理16个元素的归约
  if (blockSize >= 32)
    cache[tid] += cache[tid + 16];
  // 处理8个元素的归约
  if (blockSize >= 16)
    cache[tid] += cache[tid + 8];
  // 处理4个元素的归约
  if (blockSize >= 8)
    cache[tid] += cache[tid + 4];
  // 处理2个元素的归约
  if (blockSize >= 4)
    cache[tid] += cache[tid + 2];
  // 处理最后一个元素的归约
  if (blockSize >= 2)
    cache[tid] += cache[tid + 1];
}

/**
 * @brief 优化的归约核函数
 * @tparam blockSize 线程块大小
 * @tparam NUM_PER_THREAD 每个线程处理的元素数量
 * @param d_in 输入数据指针
 * @param d_out 输出数据指针
 * @param n 数据总长度
 * 
 * 优化策略：
 * 1. 每个线程处理多个数据元素，减少线程调度开销
 * 2. 使用共享内存进行block内归约
 * 3. warp级别优化最后32个线程
 * 4. 手动展开大blockSize的归约循环
 */
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v6(float *d_in, float *d_out, int n) {
  // 声明共享内存，用于block内线程间的数据共享
  volatile __shared__ float s_data[THREAD_PER_BLOCK];
  
  // 计算当前block处理的输入数据起始位置
  float *d_in_begin = d_in + blockIdx.x * NUM_PER_THREAD * blockSize;
  unsigned int tid = threadIdx.x;
  
  // 初始化共享内存中的累加器
  s_data[tid] = 0;
  
  // 每个线程处理NUM_PER_THREAD个元素，进行预归约
  // 这样可以显著减少后续归约的线程数量
  for (int i = 0; i < NUM_PER_THREAD; ++i)
    s_data[tid] = s_data[tid] + d_in_begin[tid + i * blockDim.x];
  
  // 同步所有线程，确保数据加载完成
  __syncthreads();

  // 在共享内存中进行归约操作
  // 手动展开不同blockSize的归约，避免运行时判断
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
  
  // 最后32个线程使用warp级别优化
  if (tid < 32)
    warpreduce<blockSize>(s_data, tid);

  // 将最终结果写回全局内存
  if (tid == 0)
    d_out[blockIdx.x] = s_data[0];
}

/**
 * @brief 结果验证函数
 * @tparam T 数据类型
 * @param out GPU计算结果
 * @param res CPU参考结果
 * @param n 数据长度
 * @return 验证是否通过
 */
template <typename T> 
bool check(T *out, T *res, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != res[i])
      return false;
  }
  return true;
}

/**
 * @brief 主函数 - 测试reduce_v6性能
 * 主要功能：
 * 1. 分配和初始化数据
 * 2. CPU参考计算
 * 3. GPU并行计算
 * 4. 结果验证
 */
int main() {
  const int N = 32 * 1024 * 1024;  // 32M个float元素
  float *a = (float *)malloc(N * sizeof(float));
  float *d_a;
  cudaMalloc((void **)&d_a, N * sizeof(float));

  // 设置block配置参数
  const int block_size = 1024; //设置block的数量
  const unsigned int NUM_PER_BLOCK = (N + block_size - 1) / block_size;
  const int NUM_PER_THREAD = (NUM_PER_BLOCK + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
  
  float *out = (float *)calloc(block_size, sizeof(float));
  float *d_out;
  cudaMalloc((void **)&d_out, block_size * sizeof(float));
  float *res = (float *)malloc(block_size * sizeof(float));

  // 初始化测试数据
  for (int i = 0; i < N; ++i) {
    a[i] = i % 100;  // 使用模100确保数据不会过大
  }

  // CPU上的规约计算 - 作为参考结果
  for (int i = 0; i < block_size; ++i) {
    float cur = 0;
    for (int j = 0; j < NUM_PER_BLOCK; j++) {
      cur += a[i * NUM_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  // 将数据从主机复制到设备
  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  // 配置CUDA执行参数
  dim3 Grid(block_size, 1);      // Grid维度
  dim3 Block(THREAD_PER_BLOCK, 1);  // Block维度
  
  // 执行GPU归约计算
  reduce_v6<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid, Block>>>(d_a, d_out, N);
  
  // 将结果从设备复制回主机
  cudaMemcpy(out, d_out, block_size * sizeof(float), cudaMemcpyDeviceToHost);
  
  // 验证计算结果
  if (check(out, res, block_size))
    printf("the ans is right\n");
  else {
    printf("the ans is wrong\n");
    for (int i = 0; i < block_size; i++) {
      printf("%lf ", out[i]);
    }
    printf("\n");
  }

  // 清理资源
  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
  free(res);
  return 0;
}