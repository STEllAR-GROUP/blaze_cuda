# Project version
NAME    = prog

# CUDA arch
CUDA_GPU_ARCH ?= sm_75

# Compilers & linker
CXX ?= clang++
CU  ?= clang++
LD  ?= clang++

# Compile flags
CXXFLAGS += -O3 -march=native

CXXFLAGS += -DBLAZE_USE_SHARED_MEMORY_PARALLELIZATION -DBLAZE_USE_HPX_THREADS \
				-DBLAZE_CUDA_MODE -DBLAZE_CUDA_USE_THRUST

CXXFLAGS += -Wall -Wextra -Werror -Wnull-dereference \
            -Wdouble-promotion -Wshadow

# Language
CXXFLAGS += -std=c++17

# Includes
INCLUDES += -I.. -I$(CUDA_HOME)/include -I/opt/thrust -I.
CXXFLAGS += $(INCLUDES)

# CUDA flags
CUFLAGS += --cuda-gpu-arch=$(CUDA_GPU_ARCH)

# Linker
LDFLAGS += -fPIC -O3
LDFLAGS += -lm -lcudart -lhpx

# Remote execution variables
REMOTE_EXEC_HOST ?= localhost
REMOTE_EXEC_PATH ?= /tmp/blaze_cuda_bench
