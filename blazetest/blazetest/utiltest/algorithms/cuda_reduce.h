//=================================================================================================
/*!
//  \file src/utiltest/alignedallocator/ClassTest.cpp
//  \brief Source file for the AlignedAllocator class test
//
//  Copyright (C) 2019 Jules Penuchot - All Rights Reserved
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

#ifndef _BLAZETEST_UTILTEST_ALGORITHMS_CUDA_REDUCE_H_
#define _BLAZETEST_UTILTEST_ALGORITHMS_CUDA_REDUCE_H_

#include <tuple>
#include <cstddef>
#include <stdexcept>

#include <blaze/Blaze.h>

#include <blaze_cuda/math/dense/CUDADynamicVector.h>
#include <blaze_cuda/util/algorithms/CUDAReduce.h>
#include <blaze_cuda/util/algorithms/Unroll.h>

namespace blazetest {

namespace utiltest {

namespace cuda_reduce {

template<typename T, std::size_t U, std::size_t B>
void propagation_test_case(std::size_t size)
{
   using std::size_t;

   // Vector type parameters
   using vtype = blaze::CUDADynamicVector<T>;

   // Number of elements computed per block
   size_t constexpr elements_per_block = U * (1 << B);

   vtype a( size, T(1) );

   for(auto const& v : a) if( v != T(1) ) {
      // TODO: Better error reporting
      throw std::runtime_error("Bad init");
   }

   // Checking that elements get reduced correctly
   auto val = blaze::cuda_reduce < U, B > ( a.begin(), a.end(), T(1)
      , [] __device__ ( T const& a, T const& b ) { return a * b; } );

   for(auto const& v : a) if( v != T(1) ) {
      // TODO: Better error reporting
      throw std::runtime_error("Altered tab");
   }

   if( val != T( 1 ) ) {
      // TODO: Better error reporting
      throw std::runtime_error( "Invalid result.\n" );
   }
}

template<typename T, std::size_t U, std::size_t B>
void count_test_case(std::size_t size)
{
   using std::size_t;

   // Vector type parameters
   using vtype = blaze::CUDADynamicVector<T>;

   // Number of elements computed per block
   size_t constexpr elements_per_block = U * (1 << B);

   vtype a( size, T(1) );

   for(auto const& v : a) if( v != T(1) ) {
      // TODO: Better error reporting
      throw std::runtime_error("Bad init");
   }

   // Checking that elements get reduced correctly
   auto val = blaze::cuda_reduce < U, B > ( a.begin(), a.end(), T(0)
      , [] __device__ ( T const& a, T const& b ) { return a + b; } );

   for(auto const& v : a) if( v != T(1) ) {
      // TODO: Better error reporting
      throw std::runtime_error("Altered tab");
   }

   if( val != T( size ) ) {
      // TODO: Better error reporting
      throw std::runtime_error( "Invalid result.\n" );
   }
}

template<typename T, std::size_t U, std::size_t B>
void test_case( std::size_t size )
{
   propagation_test_case< T, U, B >( size );
   count_test_case< T, U, B >( size );
}

template<typename T>
void launch_tests_for_type()
{
   using std::tuple;

   // Setting up kernel parameters
   auto constexpr params =
      tuple ( tuple(1 , 0) , tuple(1 , 1) , tuple(2 , 2) , tuple(2 , 3)
            , tuple(4 , 4) , tuple(8 , 8) , tuple(13, 8) , tuple(16, 8)
            , tuple(32, 8)
            );

   // Indexes for kernel parameters
   auto constexpr Unroll = 0;
   auto constexpr BlkExp = 1;

   constexpr T init( 1 );

   auto size_factors = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

   blaze::unroll< std::tuple_size< decltype( params )>::value >( [&]( auto I )
   {
      using std::get;
      auto constexpr U = get< Unroll >( get< I() >( params ) );
      auto constexpr B = get< BlkExp >( get< I() >( params ) );

      size_t constexpr elements_per_block = U * ( 1 << B );

      // Even sizes
      for( auto const& size : size_factors )
         test_case< T, U, B >( size );
      test_case< T, U, B >( elements_per_block );
      test_case< T, U, B >( elements_per_block * 2 );

      // Odd sizes
      for( auto const& size : size_factors )
         test_case< T, U, B >( size + 7 );
      test_case< T, U, B >( elements_per_block + 7 );
      test_case< T, U, B >( elements_per_block * 2 + 7 );
   } );
}

} // blazetest

} // utiltest

} // cuda_reduce

#endif
