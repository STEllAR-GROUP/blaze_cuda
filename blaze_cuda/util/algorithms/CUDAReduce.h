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

#include <cstddef>
#include <type_traits>

#include <blaze/system/CUDAAttributes.h>

#include <blaze_cuda/util/algorithms/Unroll.h>
#include <blaze_cuda/util/CUDAErrorManagement.h>

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
   void BLAZE_GLOBAL reduce_kernel ( InOutIt inout_beg, T const& init
      , BinOp const& binop = [] BLAZE_HOST_DEVICE ( auto const& a, auto const& b )
         { return a + b; } )
   {
      auto const thread_count = blockDim.x * gridDim.x;                 // Total amount of threads
      auto const thread_id    = blockDim.x * blockIdx.x + threadIdx.x;  // Global thread index

      auto a_beg = inout_beg + ( thread_id * Unroll );
      auto b_beg = inout_beg + ( thread_id * Unroll * thread_count );

      auto& out = *( inout_beg + thread_id );

      T red = init;

      unroll<Unroll>( [&]( auto )
      {
         red = binop( red, binop( *a_beg, *b_beg ) );
         a_beg++; b_beg++;
      } );

      out = binop( out, red );

      __syncthreads();
   }

}  // namespace cuda_reduce_detail

template < std::size_t Unroll = 4
         , typename InputOutputIt
         , typename T
         , typename BinOp >
inline auto cuda_reduce ( InputOutputIt inout_beg, InputOutputIt inout_end
   , T init, BinOp const& binop )
   -> decltype( *inout_beg )
{
   using std::size_t;

   //using ElmtType = std::decay_t< decltype( *inout_beg ) >;

   constexpr size_t block_size      = 128;                      // Number of threads per block
   constexpr size_t elmt_per_block  = block_size * Unroll * 2;  // Number of reduced elements per block

   auto elmt_cnt = inout_end - inout_beg; // Number of elements

   auto procecssed_by_gpu  = elmt_cnt % elmt_per_block;           // Number of elements to be processed by the GPU at each iteration
   auto block_cnt          = procecssed_by_gpu / elmt_per_block;  // Number of blocks to be launched at each iteration
   auto prev_block_cnt     = block_cnt;                           // Keeping a track of the number of blocks

   while( block_cnt > 0 )
   {
      cuda_reduce_detail::reduce_kernel < Unroll > <<< block_cnt, block_size >>>
         ( inout_beg, binop, init, binop );

      procecssed_by_gpu /= elmt_per_block;
      prev_block_cnt    = block_cnt;
      block_cnt         = procecssed_by_gpu / elmt_per_block;

      //cudaDeviceSynchronize();
   }

   auto ret = init;

   // Scalar end
   for( auto scal_beg = inout_end - ( elmt_cnt % elmt_per_block )
      ; scal_beg < inout_end
      ; scal_beg++ )
   { ret = binop( ret, *scal_beg ); }

   // Gather the rest
   cudaDeviceSynchronize();

   CUDA_ERROR_CHECK;

   auto const gath_end = inout_beg + ( prev_block_cnt * block_size );
   for( auto gath_beg = inout_beg
      ; gath_beg < gath_end
      ; gath_beg++ )
   { ret = binop( ret, *gath_beg ); }

   return ret;
}

}  // namespace blaze



#endif
