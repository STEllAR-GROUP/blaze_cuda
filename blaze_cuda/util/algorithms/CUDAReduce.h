//=================================================================================================
/*!
//  \file blaze/util/CUDAReduce.h
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

#include <numeric>
#include <cstddef>
#include <type_traits>

#include <blaze/system/CUDAAttributes.h>

#include <blaze_cuda/util/algorithms/Unroll.h>

#include <cuda_runtime.h>

namespace blaze {

namespace cuda_reduce_detail {

   // Main idea:
   //
   //       See: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
   //
   //    Implementation of the reduction algorithm described in the PDF above,
   //    but with C++ generic constructs (generic unrolling, generic binop...)

   template < std::size_t Unroll = 4
            , typename InOutIt, typename OutputIt
            , typename T
            , typename BinOp >
   void BLAZE_GLOBAL reduce_kernel ( InOutIt inout_beg, T const& init = T(0)
      , BinOp const& binop = [] BLAZE_HOST_DEVICE ( auto const& a, auto const& b )
         { return a + b; } )
   {
      using ElmtType = std::decay_t< decltype( *inout_beg ) >;

      auto const thread_count = blockDim.x * gridDim.x;                 // Total amount of threads
      auto const thread_id    = blockDim.x * blockIdx.x + threadIdx.x;  // Global thread index

      auto a_beg = inout_beg + ( thread_id * Unroll );
      auto b_beg = inout_beg + ( thread_id * Unroll * thread_count );

      auto& out = *( inout_beg + thread_id );

      auto red = init;

      unroll<Unroll>( [&] BLAZE_DEVICE ( auto const& )
      {
         red = binop( red, binop( *a_beg, *b_beg ) );
         a_beg++; b_beg++;
      } );

      out = binop( out, red );

      __syncthreads();
   }

}  // namespace cuda_reduce_detail

template < std::size_t Unroll = 4
         , typename InputIt
         , typename OutputIt
         , typename T
         , typename BinOp >
inline auto cuda_reduce ( InputIt in_beg, InputIt in_end, T init, BinOp const& binop )
   -> decltype( *in_beg )
{
   using std::size_t;

   using ElmtType = std::decay_t< decltype( *in_beg ) >;

   constexpr size_t block_size     = 128;                     // Number of threads per block
   constexpr size_t elmt_per_block = block_size * Unroll * 2; // Number of reduced elements per block

   auto elmt_count         = in_end - in_beg;                      // Number of total elements

   auto blocked_elmt_count = elmt_count % block_size;              // Number of elements to be processed by the kernel at each iteration
   auto block_count        = blocked_elmt_count / elmt_per_block;  // Number of blocks to be launched at each iteration

   while( block_count > 0 )
   {
      cuda_reduce_detail::reduce_kernel < Unroll > <<< block_count , block_size >>>
         ( in_beg, binop, init, binop );

      blocked_elmt_count /= elmt_per_block;
      block_count         = blocked_elmt_count / elmt_per_block;

      //cudaDeviceSynchronize();
   }

   // Scalar end

   std::reduce( in_beg + blocked_elmt_count, in_end, init, binop );

   cudaDeviceSynchronize();

   // Gather the rest

   return *in_beg;
}

}  // namespace blaze



#endif
