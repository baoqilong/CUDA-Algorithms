cmake_minimum_required(VERSION 3.18)
project(CUDAReduce LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA包
find_package(CUDA REQUIRED)

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 设置CUDA编译器
set(CMAKE_CUDA_COMPILER nvcc)

# 设置CUDA架构（根据你的GPU调整）
# 常见架构: 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES 75)

# 编译选项
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -lineinfo)

# 添加各个版本的可执行文件
add_executable(reduce_base
    reduce_base.cu
)

add_executable(reduce_v0
    reduce_v0.cu
)

add_executable(reduce_v1
    reduce_v1.cu
)

add_executable(reduce_v2
    reduce_v2.cu
)

add_executable(reduce_v3
    reduce_v3.cu
)

add_executable(reduce_v4
    reduce_v4.cu
)

add_executable(reduce_v5
    reduce_v5.cu
)

add_executable(reduce_v6
    reduce_v6.cu
)

# 为每个目标设置CUDA架构
set_property(TARGET reduce_base PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v0 PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v1 PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v2 PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v3 PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v4 PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v5 PROPERTY CUDA_ARCHITECTURES 75)
set_property(TARGET reduce_v6 PROPERTY CUDA_ARCHITECTURES 75)

# 链接CUDA库和线程库
foreach(target reduce_base reduce_v0 reduce_v1)
    target_link_libraries(${target} ${CUDA_LIBRARIES})
    # 如果需要chrono等C++标准库功能
    target_link_libraries(${target} pthread)
endforeach()

# 创建自定义目标用于清理所有构建文件
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build files"
)

# 添加性能分析选项（可选）
option(ENABLE_PROFILING "Enable CUDA profiling" OFF)
if(ENABLE_PROFILING)
    foreach(target reduce_base reduce_v0 reduce_v1)
        target_compile_options(${target} PRIVATE -G)
        target_link_options(${target} PRIVATE -lineinfo)
    endforeach()
endif()

# 添加调试选项（可选）
option(ENABLE_DEBUG "Enable debug mode" OFF)
if(ENABLE_DEBUG)
    foreach(target reduce_base reduce_v0 reduce_v1)
        target_compile_options(${target} PRIVATE -g -O0)
    endforeach()
endif()