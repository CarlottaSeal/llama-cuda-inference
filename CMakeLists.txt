cmake_minimum_required(VERSION 3.18)
project(llama-cuda-inference LANGUAGES CXX CUDA)

# ---- C++ / CUDA standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- GPU architecture (override with -DCUDA_ARCH=80, 86, 89, 90, 120, etc.) ----
set(CUDA_ARCH "89" CACHE STRING "CUDA SM architecture (e.g. 80=Ampere, 89=Ada, 90=Hopper, 120=Blackwell)")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})

# ---- Compiler flags ----
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# ---- Include paths ----
include_directories(${PROJECT_SOURCE_DIR}/include)

# ---- Model library (C++ sources for safetensors, tokenizer, etc.) ----
set(MODEL_SOURCES
    src/model/safetensors.cpp
    src/model/weight_loader.cpp
    src/model/tokenizer.cpp
)
add_library(llama_model STATIC ${MODEL_SOURCES})
target_compile_options(llama_model PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# .cpp sources include cuda_runtime.h; the host compiler needs the toolkit headers
find_package(CUDAToolkit REQUIRED)
target_include_directories(llama_model PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# ---- Kernel library (static) ----
# Files added incrementally per milestone
set(KERNEL_SOURCES
    # M1: element-wise kernels
    src/kernels/residual.cu
    src/kernels/silu.cu
    src/kernels/embedding.cu
    # M2: reduction kernels
    src/kernels/rmsnorm.cu
    src/kernels/softmax.cu
    # M3: RoPE
    src/kernels/rope.cu
    # M4: GEMM
    src/kernels/gemm.cu
    # M5: Attention
    src/kernels/attention.cu
)

add_library(llama_kernels STATIC ${KERNEL_SOURCES})
target_compile_options(llama_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# ---- Main executable ----
add_executable(llama_inference src/main.cu)
target_link_libraries(llama_inference llama_kernels llama_model)

# ---- Test executable ----
add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels llama_kernels)
