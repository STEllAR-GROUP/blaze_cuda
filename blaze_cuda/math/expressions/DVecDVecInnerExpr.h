//=================================================================================================
/*!
//  \file blaze_cuda/math/expressions/DVecDVecInnerExpr.h
//  \brief Header file for the dense vector/dense vector inner product expression
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_CUDA_MATH_EXPRESSIONS_DVECDVECINNEREXPR_H_
#define _BLAZE_CUDA_MATH_EXPRESSIONS_DVECDVECINNEREXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include <blaze/math/expressions/DVecDVecInnerExpr.h>
#include <blaze/math/traits/DeclSymTrait.h>
#include <blaze/math/functors/Add.h>
#include <blaze/math/functors/Mult.h>
#include <blaze/system/CUDAAttributes.h>

#include <blaze_cuda/math/dense/CUDADynamicVector.h>
#include <blaze_cuda/util/algorithms/CUDAReduce.h>
#include <blaze_cuda/util/BinopIterator.h>


namespace blaze {

template< typename ET1    // Type of the left-hand side dense vector
        , typename ET2 >  // Type of the right-hand side dense vector
inline auto dvecdvecinner( const CUDADynamicVector<ET1,true>& lhs
   , const CUDADynamicVector<ET2,false>& rhs )
{
   using blaze::BinopIterator;

   return thrust::reduce( thrust::device,
      BinopIterator( lhs.begin(), rhs.begin(), blaze::Mult() ),
      BinopIterator( lhs.end()  , rhs.end()  , blaze::Mult() ),
      ET1(0), blaze::Add() );
}

} // namespace blaze

#endif
