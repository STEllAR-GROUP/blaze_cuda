# Project version
NAME    = prog

# CUDA
CUDA_HOME     ?= /opt/cuda
CUDA_GPU_ARCH ?= sm_75

# Compilers & linker
CXX ?= clang++
CU  ?= clang++

# Compile flags
CXXFLAGS += -O3 -march=corei7

CXXFLAGS += -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION=1 \
            -DBLAZE_CUDA_MODE=1 -DBLAZE_BLAS_MODE=1 -DBLAZE_CUDA_USE_THRUST=1

CXXFLAGS += -Wall -Wextra -Werror -Wnull-dereference \
            -Wdouble-promotion -Wshadow

# Language
CXXFLAGS += -std=c++17

# Includes
INCLUDES += -I.. -I$(CUDA_HOME)/include -I/opt/thrust
CXXFLAGS += $(INCLUDES)

# CUDA flags
CUFLAGS += --cuda-gpu-arch=$(CUDA_GPU_ARCH) --cuda-path=$(CUDA_HOME)

# Linker
LDFLAGS += -fPIC -O3
LDFLAGS += -L$(CUDA_HOME)/lib64
LDFLAGS += -lm -lcudart -lhpx -lcublas -lcblas


## Development

# Remote execution variables
REMOTE_EXEC_HOST ?= localhost
REMOTE_EXEC_PATH ?= /tmp/blaze_cuda_dev
