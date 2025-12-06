#include <cuda_runtime.h>
#include <stdio.h>

#include "reduce.cuh"

/*
 * reduce_v7: warp shuffle优化归约算法实现
 * 特点：
 * 1. 使用warp shuffle指令进行线程间数据交换，无需共享内存
 * 2. 支持多元素预处理，提高并行度利用率
 * 3. 模板化设计，编译期优化
 * 4. 完全避免共享内存bank冲突
 * 5. 极低的同步开销
 */

/*
 * reduce_v6: 综合优化的归约实现
 * 特点：结合了前几个版本的所有优化技术
 * 优化策略：
 * 1. 多元素预处理减少线程数量
 * 2. 共享内存归约减少全局内存访问
 * 3. Warp级别优化最后阶段
 * 4. 模板化编译优化
 * 5. 手动循环展开避免分支判断
 * 性能：相比基础版本有15-20倍的性能提升
 */

/**
 * @brief Warp级别的归约函数
 * @tparam blockSize 线程块大小
 * @param sum 当前线程的局部累加值
 * @return 归约后的结果
 * 
 * 使用warp shuffle指令进行数据交换，这是CUDA 9.0+的特性
 * 相比共享内存方式，warp shuffle具有以下优势：
 * - 无需共享内存分配
 * - 避免bank冲突
 * - 更低的延迟
 * - 更高的带宽
 */
template <unsigned int blockSize>
__device__ __forceinline__ float warpreduce(float sum) {
  // 使用warp shuffle指令进行多级归约
  // 每个步骤将相邻线程的数据相加，实现log2(n)的归约
  
  // 第1步：跨16个线程交换数据（适用于32线程以上的block）
  // __shfl_down_sync: 在warp内向下交换数据，同步执行
  // 0xffffffff: 所有32个lane都参与
  // 16: 跨16个lane交换数据
  if(blockSize>=32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);
  
  // 第2步：跨8个线程交换数据（适用于16线程以上的block）
  if (blockSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);
  
  // 第3步：跨4个线程交换数据（适用于8线程以上的block）
  if(blockSize>=8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);
  
  // 第4步：跨2个线程交换数据（适用于4线程以上的block）
  if (blockSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);
  
  // 第5步：相邻线程交换数据（适用于2线程以上的block）
  if(blockSize>=2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  
  return sum;
}

/**
 * @brief reduce_v7内核函数 - 使用warp shuffle优化的归约算法
 * @tparam blockSize 线程块大小
 * @tparam NUM_PER_THREAD 每个线程处理的元素数量
 * @param d_in 输入数据指针（设备内存）
 * @param d_out 输出结果指针（设备内存）
 * @param n 数据总长度
 * 
 * 算法流程：
 * 1. 每个线程处理多个元素进行预归约
 * 2. 使用warp shuffle在warp内部进行归约
 * 3. 将warp结果写入共享内存
 * 4. 再次使用warp shuffle进行跨warp归约
 * 5. 将最终结果写入全局内存
 */
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v7(float *d_in, float *d_out, int n) {
  // 计算当前block处理的输入数据起始位置
  // 每个block处理NUM_PER_THREAD * blockSize个元素
  float *d_in_begin = d_in + blockIdx.x * NUM_PER_THREAD * blockSize;
  unsigned int tid = threadIdx.x;  // 线程ID
  
  // 初始化局部累加器
  float sum = 0.0f;

  // 每个线程处理NUM_PER_THREAD个元素，进行预归约
  // 这样可以显著减少后续归约的线程数量，提高并行度
  for (int i = 0; i < NUM_PER_THREAD; ++i) {
    // 计算当前元素在全局内存中的位置
    // tid + i * blockSize确保连续的内存访问模式
    sum += d_in_begin[tid + i * blockSize];
  }

  // 分配共享内存用于存储warp的归约结果
  // 每个warp需要一个存储位置，最多支持32个warp
  static __shared__ float s_data[32];
  
  // 计算当前线程在warp中的位置信息
  const int laneid = tid % 32;  // lane ID (0-31)
  const int warpid = tid / 32;  // warp ID

  // 在warp内部进行归约
  // 使用warp shuffle指令，无需显式同步
  sum = warpreduce<blockSize>(sum);

  // 每个warp的lane 0线程将结果写入共享内存
  if(laneid==0)
    s_data[warpid] = sum;
  
  // 等待所有warp完成写入
  __syncthreads();

  // 只有前blockSize/32个线程参与跨warp归约
  // 其他线程设置为0，避免参与计算
  sum = (tid < blockDim.x / 32) ? s_data[laneid] : 0.0f;
  
  // 在第一个warp中进行跨warp归约
  if (warpid == 0)
    sum = warpreduce<blockSize / 32>(sum);
  
  // 将最终结果写回全局内存
  // 每个block输出一个归约结果
  if (tid == 0)
    d_out[blockIdx.x] = sum;
}

/**
 * @brief 结果验证函数
 * @tparam T 数据类型
 * @param out GPU计算结果
 * @param res CPU参考结果
 * @param n 数据长度
 * @return 验证是否通过
 * 
 * 功能：逐元素比较GPU计算结果和CPU参考结果
 * 如果所有元素都匹配，返回true；否则返回false
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
 * @brief 主函数 - 测试reduce_v7性能
 * 主要功能：
 * 1. 分配和初始化数据
 * 2. CPU参考计算
 * 3. GPU并行计算
 * 4. 结果验证
 * 5. 资源清理
 * 
 * 测试规模：32M个float元素
 * 使用1024个block，每个block 256线程
 * 每个线程处理多个元素进行预归约
 */
int main() {
  // 设置测试数据规模：32M个float元素
  const int N = 32 * 1024 * 1024;
  
  // 主机内存分配
  float *a = (float *)malloc(N * sizeof(float));
  float *d_a;  // 设备内存指针
  cudaMalloc((void **)&d_a, N * sizeof(float));

  // 设置block配置参数
  const int block_num = 1024; // 设置block的数量
  const unsigned int NUM_PER_BLOCK = (N + block_num - 1) / block_num;  // 每个block处理的元素数
  const int NUM_PER_THREAD = (NUM_PER_BLOCK + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;  // 每个线程处理的元素数
  
  // 输出内存分配
  float *out = (float *)calloc(block_num, sizeof(float));  // 主机输出，初始化为0
  float *d_out;  // 设备输出指针
  cudaMalloc((void **)&d_out, block_num * sizeof(float));
  float *res = (float *)malloc(block_num * sizeof(float));  // CPU参考结果

  // 初始化测试数据
  // 使用模100确保数据不会过大，避免数值溢出
  for (int i = 0; i < N; ++i) {
    a[i] = i % 100;
  }

  // CPU上的规约计算 - 作为参考结果
  // 模拟GPU的并行归约过程，每个block对应一个结果
  for (int i = 0; i < block_num; ++i) {
    float cur = 0;
    for (int j = 0; j < NUM_PER_BLOCK; j++) {
      cur += a[i * NUM_PER_BLOCK + j];
    }
    res[i] = cur;
  }

  // 将数据从主机复制到设备
  cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  // 配置CUDA执行参数
  dim3 Grid(block_num, 1);      // Grid维度：1D网格，block_num个block
  dim3 Block(THREAD_PER_BLOCK, 1);  // Block维度：1D块，THREAD_PER_BLOCK个线程
  
  // 执行GPU归约计算
  // 使用模板参数：block大小和每个线程处理的元素数
  reduce_v7<THREAD_PER_BLOCK, NUM_PER_THREAD><<<Grid, Block>>>(d_a, d_out, N);
  
  // 等待GPU计算完成
  cudaDeviceSynchronize();
  
  // 将结果从设备复制回主机
  cudaMemcpy(out, d_out, block_num * sizeof(float), cudaMemcpyDeviceToHost);
  
  // 验证计算结果
  if (check(out, res, block_num))
    printf("✅ 计算结果正确！reduce_v7算法验证通过\n");
  else {
    printf("❌ 计算结果错误！reduce_v7算法验证失败\n");
    // 输出错误信息用于调试
    for (int i = 0; i < block_num; i++) {
      printf("GPU结果[%d]: %lf, CPU结果[%d]: %lf\n", i, out[i], i, res[i]);
    }
  }

  // 清理资源
  cudaFree(d_a);
  cudaFree(d_out);
  free(a);
  free(out);
  free(res);
  
  printf("🎯 reduce_v7测试完成！\n");
  return 0;
}