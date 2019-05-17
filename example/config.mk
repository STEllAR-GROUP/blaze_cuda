# Project version
NAME    = prog
VERSION = 0.0.1

# Paths
PREFIX    ?= /usr/local
MANPREFIX ?= $(PREFIX)/share/man

# CUDA arch
CUDA_GPU_ARCH ?= sm_75

# Compiler
CXX ?= clang++
CU  ?= clang++

# Compile flags
CXXFLAGS += -O3 -march=native
CXXFLAGS += -DVERSION=\"$(VERSION)\"
CXXFLAGS += -Wall -Wextra -Werror -Wnull-dereference \
            -Wdouble-promotion -Wshadow

# Language
CXXFLAGS += -std=c++17

# Includes
INCLUDES += -I.. -I$(CUDA_HOME)/include
CXXFLAGS += $(INCLUDES)

# CUDA flags
CUFLAGS += --cuda-gpu-arch=$(CUDA_GPU_ARCH)

# Linker
LDFLAGS += -fPIC -O3
LDFLAGS += -lm -lcudart


## Development

# Remote execution variables
REMOTE_EXEC_HOST ?= localhost
REMOTE_EXEC_PATH ?= /tmp/blaze_cuda_dev
