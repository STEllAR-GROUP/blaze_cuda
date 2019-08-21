//=================================================================================================
/*!
//  \file blaze_cuda/math/expressions/DMatDMatMultExpr.h
//  \brief Header file for the dense matrix/dense matrix multiplication expression
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

#ifndef _BLAZE_CUDA_MATH_EXPRESSIONS_DMATDMATMULTEXPR_H_
#define _BLAZE_CUDA_MATH_EXPRESSIONS_DMATDMATMULTEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DMatDMatMultExpr.h>
#include <blaze/math/traits/DeclSymTrait.h>

#include <blaze_cuda/math/cublas/gemm.h>
#include <blaze_cuda/math/typetraits/RequiresCUDAEvaluation.h>

namespace blaze {

template < typename MT   // Type of the target dense matrix
         , bool SO       // Storage order of the target dense matrix
         , typename MT1  // Type of the left-hand side dense matrix
         , typename MT2  // Type of the right-hand side dense matrix
         , bool SF       // Symmetry flag
         , bool HF       // Hermitian flag
         , bool LF       // Lower flag
         , bool UF >     // Upper flag
inline auto cudaAssign( DenseMatrix<MT,SO>& lhs, const DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

   if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
      return;
   }
   else if( rhs.leftOperand().columns() == 0UL ) {
      reset( ~lhs );
      return;
   }

   using ExpType = DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF>;
   using ET = typename MT::ElementType;
   using LT = typename ExpType::LT;
   using RT = typename ExpType::RT;

   LT A( rhs.leftOperand() );    // Evaluation of the left-hand side dense matrix operand
   RT B( rhs.rightOperand() );   // Evaluation of the right-hand side dense matrix operand

   BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.leftOperand().rows()    , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( A.columns() == rhs.leftOperand().columns() , "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rightOperand().rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rightOperand().columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

   cugemm( ~lhs, ~A, ~B, ET(1), ET(0) );
}

template < typename MT   // Type of the target dense matrix
         , bool SO       // Storage order of the target dense matrix
         , typename MT1  // Type of the left-hand side dense matrix
         , typename MT2  // Type of the right-hand side dense matrix
         , bool SF       // Symmetry flag
         , bool HF       // Hermitian flag
         , bool LF       // Lower flag
         , bool UF >     // Upper flag
inline auto cudaAddAssign( DenseMatrix<MT,SO>& lhs, const DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

   if( (~lhs).rows() == 0UL || (~lhs).columns() == 0UL ) {
      return;
   }
   else if( rhs.leftOperand().columns() == 0UL ) {
      reset( ~lhs );
      return;
   }

   using ExpType = DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF>;
   using ET = typename MT::ElementType;
   using LT = typename ExpType::LT;
   using RT = typename ExpType::RT;

   LT A( rhs.leftOperand() );    // Evaluation of the left-hand side dense matrix operand
   RT B( rhs.rightOperand() );   // Evaluation of the right-hand side dense matrix operand

   BLAZE_INTERNAL_ASSERT( A.rows()    == rhs.leftOperand().rows()    , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( A.columns() == rhs.leftOperand().columns() , "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( B.rows()    == rhs.rightOperand().rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( B.columns() == rhs.rightOperand().columns(), "Invalid number of columns" );
   BLAZE_INTERNAL_ASSERT( A.rows()    == (~lhs).rows()     , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( B.columns() == (~lhs).columns()  , "Invalid number of columns" );

   cugemm( ~lhs, A, B, ET(1), ET(1) );
}

template < typename MT1, typename MT2, bool SF, bool HF, bool LF, bool UF >
struct RequiresCUDAEvaluation< DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF>
   , EnableIf_t< IsCUDAAssignable_v< DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF> > > >
{
public:
   static constexpr bool value = true;
};

} // namespace blaze

#endif
