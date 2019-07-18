# Blaze CUDA · WIP

CUDA extension for [Blaze](https://bitbucket.org/blaze-lib/blaze).

## Introduction

The library is made to add CUDA capability to Blaze by adding CUDA vector, matrix and tensor types.

## Build requirements

The only requirement is to use `clang` in CUDA mode instead of `nvcc`. `nvcc` fails to compile Blaze despite being "C++14-compatible", whereas `clang` succeeds in CUDA mode. Additionally, `clang` outputs cleaner error messages and provides a more standard shell interface, which makes scripting, and dependency management in makefiles easier.

The `example` folder provides a simple `Makefile` that can be used as a reference for projects that use Blaze CUDA.

## Roadmap (expected to change)

* [X] Implementing a base CUDA allocated vector type
* [X] Base assign function implementation for vector type
* [X] Extending Blaze's type_trait library to integrate CUDAManagedVector into Blaze
* [X] Adding IsCUDAEnabled conditional specializations to `DVecDVec*Expr`
* [X] Adapting basic expression iterators for vectors
* [X] Adapting functors to make them CUDA-compatible
* [X] **cuBLAS**
* [X] Cleanup: Refactor CUDADynamicVector's code (Indent, documentation, etc.)
* [X] Extending functionality to matrices
* [ ] Adapting fancier expressions (reductions, etc)
* [ ] Getting Matrix/Vector operations to work
* [ ] Extending functionality to tensors
