# CUDA arch
CUDA_GPU_ARCH ?= sm_70

# Compiler
CXX ?= clang++
CU  ?= clang++

# Compile flags
CXXFLAGS += -O3 -march=native
CXXFLAGS += -DVERSION=\"$(VERSION)\"
# CXXFLAGS += -Wall -Wextra -Werror -Wnull-dereference \
#             -Wdouble-promotion -Wshadow

# Language
CXXFLAGS += -std=c++17

# Includes
INCLUDES += -Iinclude -I$(CUDA_HOME)/include -I. -I../
CXXFLAGS += $(INCLUDES)

# CUDA flags
CUFLAGS += --cuda-gpu-arch=$(CUDA_GPU_ARCH)

# Linker
LDFLAGS += -fPIC -lm -lcudart
