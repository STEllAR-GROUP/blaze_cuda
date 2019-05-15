//=================================================================================================
/*!
//  \file blaze/util/algorithms/CUDAReduce.h
//  \brief Header file for CUDAReduce's implementation
//
//  Copyright (C) 2012-2019 Jules Pénuchot - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_CUDA_UTIL_ALGORITHMS_CUDAREDUCE_H_
#define _BLAZE_CUDA_UTIL_ALGORITHMS_CUDAREDUCE_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include <iostream>

#include <blaze_cuda/util/algorithms/Unroll.h>
#include <blaze_cuda/util/algorithms/CUDATransform.h>
#include <blaze_cuda/util/CUDAErrorManagement.h>

#include <cuda_runtime.h>

namespace blaze {

namespace cuda_reduce_detail {

   // Util function
   constexpr std::size_t pow2(std::size_t n) { return 2 << n; }

   // Main idea:
   //
   //       See: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   //
   //    Implementation of the reduction algorithm described in the PDF above,
   //    but with C++ generic constructs (generic unrolling, generic binop...)

template < std::size_t Unroll, std::size_t BlockSizeExponent
         , typename InputIt , typename OutputIt
         , typename T
         , typename BinOp >
void __global__ reduce_kernel ( InputIt in_beg, OutputIt out_beg, T init, BinOp binop )
{
   constexpr size_t block_size = pow2( BlockSizeExponent );

   // Shared memory pool
   __shared__ std::array<T, block_size * 2> sdata;

   // Thread indexing
   auto const thread_count = blockDim.x * gridDim.x;
   auto const global_id    = blockDim.x * blockIdx.x + threadIdx.x;

   // Array indexing
   auto begin = in_beg + ( global_id * Unroll );

   // Accumulator init
   T acc = init;

   // Computation, unrolled N times
   unroll< Unroll >( [&]( auto const& I ) {
      acc = binop( acc, *( begin + ( I() * thread_count ) ) );
   } );

   auto& self = sdata[ threadIdx.x ];

   // Storing result
   self = binop( self, acc );

   __syncthreads();

   // Block reduction
   unroll< BlockSizeExponent >( [&] ( auto I ) {
      auto constexpr Delta = pow2( BlockSizeExponent - I() - 1 );
      self = sdata[ threadIdx.x + Delta ];
      __syncthreads();
   } );

   // Storing result
   __syncthreads();
   if( threadIdx.x == 0 ) *( out_beg + blockIdx.x ) = sdata[ 0 ];
}

}  // namespace cuda_reduce_detail


template < std::size_t Unroll = 4, std::size_t BlockSizeExponent = 8
         , typename InputOutputIt
         , typename T
         , typename BinOp >
inline auto cuda_reduce ( InputOutputIt inout_beg, InputOutputIt inout_end, T init, BinOp binop )
{
   using cuda_reduce_detail::reduce_kernel;
   using cuda_reduce_detail::pow2;

   using std::size_t;
   constexpr size_t block_size = pow2( BlockSizeExponent );
   constexpr size_t elmts_per_block = block_size * Unroll;

   // Computing

   // TODO: Second loop for max block size

   while( inout_end - inout_beg >= (ptrdiff_t)elmts_per_block )
   {
      size_t const size            = inout_end - inout_end;
      size_t const unpadded_size   = size % elmts_per_block;
      size_t const padded_size     = size - unpadded_size;
      size_t const block_cnt       = padded_size / elmts_per_block;

      blaze::cuda_zip_transform( inout_end - unpadded_size, inout_end, inout_beg, inout_beg, binop );

      cudaDeviceSynchronize();
      CUDA_ERROR_CHECK;

      reduce_kernel< Unroll, BlockSizeExponent > <<< block_cnt, block_size >>>
         ( inout_beg, inout_end, init, binop );
      inout_end = inout_beg + block_cnt;
   }

   T ret = init;

   while( inout_beg < inout_end ) ret = binop(ret, *( inout_beg++ ));

   return ret;
}


}  // namespace blaze



#endif
