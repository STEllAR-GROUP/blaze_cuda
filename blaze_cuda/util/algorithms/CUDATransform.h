//=================================================================================================
/*!
//  \file blaze_cuda/util/CUDATransform.h
//  \brief Header file for CUDATransform's implementation
//
//  Copyright (C) 2019 Jules P�nuchot - All Rights Reserved
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

#ifndef _BLAZE_CUDA_UTIL_ALGORITHMS_CUDATRANSFORM_H_
#define _BLAZE_CUDA_UTIL_ALGORITHMS_CUDATRANSFORM_H_

#include <cstddef>

#include <blaze_cuda/util/algorithms/Unroll.h>

#ifndef BLAZE_CUDA_NO_THRUST
#  include <thrust/transform.h>
#  include <thrust/execution_policy.h>
#endif

namespace blaze {

#ifndef BLAZE_CUDA_NO_THRUST

namespace detail {

template< typename IteratorType >
class ThrustInputIteratorAdapter
{
   IteratorType it;

public:

   //**Type definitions****************************************************************************
   using IteratorCategory = typename IteratorType::IteratorCategory; //!< The iterator category.
   using ValueType        = typename IteratorType::ValueType;        //!< Type of the underlying elements.
   using PointerType      = typename IteratorType::PointerType;      //!< Pointer return type.
   using ReferenceType    = typename IteratorType::ReferenceType;    //!< Reference return type.
   using DifferenceType   = typename IteratorType::DifferenceType;   //!< Difference between two iterators.

   // STL iterator requirements
   using iterator_category = IteratorCategory;  //!< The iterator category.
   using value_type        = ValueType;         //!< Type of the underlying elements.
   using pointer           = PointerType;       //!< Pointer return type.
   using reference         = ReferenceType;     //!< Reference return type.
   using difference_type   = DifferenceType;    //!< Difference between two iterators.

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE
      ThrustInputIteratorAdapter( IteratorType const& it ): it(it) {}

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE ValueType operator[]( ptrdiff_t inc ) const noexcept {
      return *ThrustInputIteratorAdapter( it + inc );
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE ValueType operator*() const noexcept {
      return *it;
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE auto
      operator-( ThrustInputIteratorAdapter const& other ) const noexcept
   {
      return it - other.it;
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE auto operator+( ptrdiff_t inc ) const noexcept {
      return ThrustInputIteratorAdapter( IteratorType( it + inc ) );
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE auto operator-( ptrdiff_t dec ) const noexcept {
      return ThrustInputIteratorAdapter( IteratorType( it - dec ) );
   }
};

template< typename IteratorType >
class ThrustOutputIteratorAdapter
{
   IteratorType it;

public:

   //**Type definitions****************************************************************************
   using IteratorCategory = typename IteratorType::IteratorCategory; //!< The iterator category.
   using ValueType        = typename IteratorType::ValueType;        //!< Type of the underlying elements.
   using PointerType      = typename IteratorType::PointerType;      //!< Pointer return type.
   using ReferenceType    = typename IteratorType::ReferenceType;    //!< Reference return type.
   using DifferenceType   = typename IteratorType::DifferenceType;   //!< Difference between two iterators.

   // STL iterator requirements
   using iterator_category = IteratorCategory;  //!< The iterator category.
   using value_type        = ValueType;         //!< Type of the underlying elements.
   using pointer           = PointerType;       //!< Pointer return type.
   using reference         = ReferenceType;     //!< Reference return type.
   using difference_type   = DifferenceType;    //!< Difference between two iterators.

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE ThrustOutputIteratorAdapter( IteratorType const& it )
      : it(it) {}

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE ReferenceType operator[]( ptrdiff_t inc ) const noexcept {
      return *ThrustOutputIteratorAdapter( it + inc );
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE ReferenceType operator*() const noexcept {
      return *it;
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE auto
      operator-( ThrustOutputIteratorAdapter const& other ) noexcept
   {
      return ThrustOutputIteratorAdapter( it - other.it );
   }

   BLAZE_ALWAYS_INLINE BLAZE_DEVICE_CALLABLE auto operator+( ptrdiff_t inc ) noexcept {
      return ThrustOutputIteratorAdapter( IteratorType( it + inc ) );
   }
};

}  // namespace detail

template < std::size_t Unroll = 16
         , typename InputIt1, typename InputIt2, typename OutputIt
         , typename F >
inline void cuda_transform ( InputIt1 in1_begin , InputIt1 in1_end
                           , InputIt2 in2_begin
                           , OutputIt out_begin
                           , F f )
{
   using namespace detail;
   using AI1 = ThrustInputIteratorAdapter<InputIt1>;
   using AI2 = ThrustInputIteratorAdapter<InputIt2>;
   using AO = ThrustOutputIteratorAdapter<OutputIt>;

   thrust::transform( thrust::device,
      AI1( in1_begin ), AI1( in1_end ),   // Meant to be the left-hand side
      AI2( in2_begin ),                   // Adaptor for the right-hand side
      AO( out_begin ), f );
}

template < std::size_t Unroll = 16
         , typename InputIt1, typename OutputIt
         , typename F >
inline void cuda_transform ( InputIt1 in1_begin , InputIt1 in1_end
                           , OutputIt out_begin
                           , F f )
{
   using namespace detail;
   using AI1 = ThrustInputIteratorAdapter<InputIt1>;

   thrust::transform( thrust::device, AI1(in1_begin), AI1(in1_end), out_begin, f );
}

#else // ifndef BLAZE_CUDA_NO_THRUST

namespace detail {

   template < std::size_t Unroll
            , typename InputIt
            , typename OutputIt
            , typename F >
   void __global__ _cuda_transform_impl( InputIt in_begin, OutputIt out_begin, F f )
   {
      using std::size_t;

      size_t const grid_size = gridDim.x * blockDim.x;
      size_t const global_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

      const auto n_in_begin  = in_begin  + global_id;
      const auto n_out_begin = out_begin + global_id;

      unroll<Unroll>( [&] ( auto const& I ) {
         auto const delta = I() * grid_size;
         *(n_out_begin + delta) = f( *(n_in_begin + delta) );
      } );
   }

   template < std::size_t Unroll
            , typename InputIt1
            , typename InputIt2
            , typename OutputIt
            , typename F >
   void __global__ _cuda_transform_impl( InputIt1 in1_begin, InputIt2 in2_begin
                                       , OutputIt out_begin, F f )
   {
      using std::size_t;

      size_t const grid_size = gridDim.x * blockDim.x;
      size_t const global_id = ((blockIdx.x * blockDim.x) + threadIdx.x);

      const auto n_in1_begin  = in1_begin  + global_id;
      const auto n_in2_begin  = in2_begin  + global_id;
      const auto n_out_begin  = out_begin  + global_id;

      unroll<Unroll>( [&] ( auto const& I ) {
         auto delta = I() * grid_size;
         *(n_out_begin + delta) = f ( *( n_in1_begin + delta )
                                    , *( n_in2_begin + delta ) );
      } );
   }

   template< std::size_t Unroll = 4, typename InputIt, typename OutputIt, typename F >
   inline void cuda_transform ( InputIt in_begin, InputIt in_end
                              , OutputIt out_begin
                              , F const& f )
   {
      using std::size_t;

      constexpr size_t max_block_size   = 512;
      constexpr size_t max_block_cnt    = 8192;
      constexpr size_t elmts_per_block  = max_block_size * Unroll;

      while( in_end - in_begin >= ptrdiff_t( elmts_per_block ) )
      {
         size_t const elmt_cnt = in_end - in_begin;
         size_t const block_cnt = elmt_cnt / elmts_per_block;

         auto const final_block_cnt = std::min( block_cnt, max_block_cnt );
         detail::_cuda_transform_impl
            <Unroll>
            <<< final_block_cnt, max_block_size >>>
            ( in_begin, out_begin, f );

         auto const incr = final_block_cnt * elmts_per_block;

         in_begin  += incr;
         out_begin += incr;
      }

      while( in_end - in_begin > 0 )
      {
         auto const final_block_size = std::min( max_block_size, size_t( in_end - in_begin ) );

         detail::_cuda_transform_impl<1> <<< 1, final_block_size >>>
            ( in_begin, out_begin, f );

         auto const incr = final_block_size;
         in_begin  += incr;
         out_begin += incr;
      }
   }

   template < std::size_t Unroll = 4
            , typename InputIt1
            , typename InputIt2
            , typename OutputIt
            , typename F >
   inline void cuda_transform ( InputIt1 in1_begin
                              , InputIt1 in1_end
                              , InputIt2 in2_begin
                              , OutputIt out_begin
                              , F const& f )
   {
      using std::size_t;

      constexpr size_t max_block_size  = 512;
      constexpr size_t max_block_cnt   = 8192;
      constexpr size_t elmts_per_block = max_block_size * Unroll;

      while( in1_end - in1_begin >= ptrdiff_t( elmts_per_block ) )
      {
         size_t const elmt_cnt = in1_end - in1_begin;
         size_t const block_cnt = elmt_cnt / elmts_per_block;

         auto const final_block_cnt = std::min( block_cnt, max_block_cnt );

         detail::_cuda_transform_impl
            <Unroll>
            <<< final_block_cnt, max_block_size >>>
            ( in1_begin, in2_begin, out_begin, f );

         auto const incr = final_block_cnt * elmts_per_block;

         in1_begin += incr;
         in2_begin += incr;
         out_begin += incr;
      }

      while( in1_end - in1_begin > 0 )
      {
         auto const final_block_size = std::min( max_block_size, size_t( in1_end - in1_begin ) );

         detail::_cuda_transform_impl
            <1>
            <<< 1, final_block_size >>>
            ( in1_begin, in2_begin, out_begin, f );

         auto const incr = final_block_size;

         in1_begin += incr;
         in2_begin += incr;
         out_begin += incr;
      }
   }
}  // namespace detail

template < std::size_t Unroll = 16
         , typename InputIt1, typename InputIt2, typename OutputIt
         , typename F >
inline void cuda_transform ( InputIt1 in1_begin , InputIt1 in1_end
                           , InputIt2 in2_begin
                           , OutputIt out_begin
                           , F f )
{
   detail::cuda_transform( in1_begin, in1_end, in2_begin, out_begin, f );
}

template < std::size_t Unroll = 16
         , typename InputIt1, typename OutputIt
         , typename F >
inline void cuda_transform ( InputIt1 in1_begin , InputIt1 in1_end
                           , OutputIt out_begin
                           , F f )
{
   detail::cuda_transform( in1_begin, in1_end, out_begin, f );
}

#endif   // ifndef BLAZE_CUDA_NO_THRUST

}  // namespace blaze

#endif
