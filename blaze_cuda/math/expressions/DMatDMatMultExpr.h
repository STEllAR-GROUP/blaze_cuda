//=================================================================================================
/*!
//  \file blaze/math/expressions/DMatDMatMultExpr.h
//  \brief Header file for the dense matrix/dense matrix multiplication expression
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#include <blaze/math/traits/DeclSymTrait.h>

#include <blaze_cuda/math/dense/CUDADynamicMatrix.h>
#include <blaze_cuda/math/cublas/gemm.h>

namespace blaze {

//**BLAS-based assignment to dense matrices (default)*******************************************
   /*!\brief Default assignment of a scaled dense matrix-dense matrix multiplication
   //        (\f$ C=s*A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \param scalar The scaling factor.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a large scaled
   // dense matrix-dense matrix multiplication expression to a dense matrix.
   */
   template< typename ET1    // Type of the left-hand side target matrix
           , typename ET2    // Type of the left-hand side matrix operand
           , typename ET3 >  // Type of the right-hand side matrix operand
   static inline auto selectBlasAssignKernel( CUDADynamicMatrix<ET1>& C,
      const CUDADynamicMatrix<ET2>& A, const CUDADynamicMatrix<ET3>& B )
   {
      static_assert(std::is_same_v<ET1, ET2> && std::is_same_v<ET2, ET3>, "message");

      gemm( C, A, B, ET1(1), ET1(0) );
   }
   //**********************************************************************************************

   //**Default assignment to dense matrices (small matrices)***************************************
   /*! \cond BLAZE_INTERNAL */
   /*!\brief Default assignment of a small dense matrix-dense matrix multiplication (\f$ C=A*B \f$).
   // \ingroup dense_matrix
   //
   // \param C The target left-hand side dense matrix.
   // \param A The left-hand side multiplication operand.
   // \param B The right-hand side multiplication operand.
   // \return void
   //
   // This function relays to the default implementation of the assignment of a dense matrix-
   // dense matrix multiplication expression to a dense matrix.
   */
   template< typename ET1    // Type of the left-hand side target matrix
           , typename ET2    // Type of the left-hand side matrix operand
           , typename ET3 >  // Type of the right-hand side matrix operand
   static inline auto selectSmallAssignKernel( CUDADynamicMatrix<ET1>& C,
      const CUDADynamicMatrix<ET2>& A, const CUDADynamicMatrix<ET3>& B )
   {
      selectBlasAssignKernel( C, A, B );
   }
   /*! \endcond */
   //**********************************************************************************************

template < typename ET   // Type of the target dense matrix
         , bool SO       // Storage order of the target dense matrix
         , typename MT1  // Type of the left-hand side dense matrix
         , typename MT2  // Type of the right-hand side dense matrix
         , bool SF       // Symmetry flag
         , bool HF       // Hermitian flag
         , bool LF       // Lower flag
         , bool UF >     // Upper flag
   inline auto smpAssign( CUDADynamicMatrix<ET,SO>& lhs, const DMatDMatMultExpr<MT1,MT2,SF,HF,LF,UF>& rhs )
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

      selectBlasAssignKernel( lhs, A, B );
   }

} // namespace blaze

#endif
