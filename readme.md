# Blaze CUDA · WIP

CUDA extension for [Blaze](https://bitbucket.org/blaze-lib/blaze).

## Introduction

The library is made to add CUDA capability to Blaze by adding CUDA vector, matrix and tensor types.

## Build requirements

The only requirement is to use `clang` in CUDA mode instead of `nvcc`. `nvcc` fails to compile Blaze despite being "C++14-compatible", whereas `clang` succeeds in CUDA mode. Additionally, `clang` output cleaner error messages and provides a more standard shell interface, which makes scripting, and dependency management in makefiles easier.

The `example` folder provides a simple `Makefile` that can be used as a reference for projects that use Blaze CUDA.

## Roadmap (expected to change)

* [X] Implementing a base CUDA allocated vector type
* [X] Base assign function implementation for vector type
* [ ] Extending Blaze's type_trait library to support CUDA
* [ ] Annotating existing Blaze iterators to support CUDA
* [ ] Extending functionality to matrices
* [ ] Extending functionality to tensors
