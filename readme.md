# Blaze CUDA · WIP

CUDA extension for [Blaze](https://bitbucket.org/blaze-lib/blaze).

## Introduction

The library is made to add CUDA capability to Blaze by adding CUDA vector, matrix and tensor types.

## Build requirements

The only requirement is to use `clang` in CUDA mode instead of `nvcc`. `nvcc` fails to compile Blaze despite being "C++14-compatible", whereas `clang` succeeds in CUDA mode. Additionally, `clang` outputs cleaner error messages and provides a more standard shell interface, which makes scripting, and dependency management in makefiles easier.

The `example` folder provides a simple `Makefile` that can be used as a reference for projects that use Blaze CUDA.

## Installation

`sudo make install`

*Uninstall target available as well*

## Features

* Dense Vectors
* Dense Matrices (no CustomMatrix yet)
* Element-wise operations for dense matrices & vectors
* [WIP] Partial cuBLAS implementation for more complex operations

[Blaze Tensor](https://github.com/stellar-group/blaze_tensor) will be supported in the future.
