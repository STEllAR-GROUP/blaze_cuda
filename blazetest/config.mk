# CUDA arch
CUDA_GPU_ARCH ?= sm_70

# Compiler
CXX ?= clang++
CU  ?= clang++

# Compile flags
CXXFLAGS += -O1 -march=native

# Language
CXXFLAGS += -std=c++17

# Includes
INCLUDES += -Iinclude -I$(CUDA_HOME)/include -I. -I../
CXXFLAGS += $(INCLUDES)

# CUDA flags
CUFLAGS += --cuda-gpu-arch=$(CUDA_GPU_ARCH)

# Linker
LDFLAGS += -fPIC -lm -lcudart
