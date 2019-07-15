//=================================================================================================
/*!
//  \file blaze/math/expressions/DMatDMatAddExpr.h
//  \brief Header file for the dense matrix/dense matrix multiplication expression
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

#ifndef _BLAZE_CUDA_MATH_EXPRESSIONS_DMATDMATDDEXPR_H_
#define _BLAZE_CUDA_MATH_EXPRESSIONS_DMATDMATDDEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DMatDMatAddExpr.h>
#include <blaze/math/traits/DeclSymTrait.h>

namespace blaze {

//**Assignment to dense matrices****************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment of a dense matrix-dense matrix addition to a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side addition expression to be assigned.
// \return void
//
// This function implements the performance optimized assignment of a dense matrix-dense
// matrix addition expression to a dense matrix. Due to the explicit application of the
// SFINAE principle, this function can only be selected by the compiler in case either
// of the two operands requires an intermediate evaluation.
*/
template< typename MT  // Type of the target dense matrix
        , bool SO2
        , typename MT1
        , typename MT2
        , bool SO >   // Storage order of the target dense matrix
inline auto cudaAssign( DenseMatrix<MT,SO2>& lhs, const DMatDMatAddExpr<MT1,MT2,SO>& rhs )
   -> EnableIf_t< DMatDMatAddExpr<MT1,MT2,SO>::useAssign >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == rhs.rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == rhs.columns(), "Invalid number of columns" );

   if( !IsOperation_v<MT1> && isSame( ~lhs, rhs.leftOperand() ) ) {
      cudaAddAssign( ~lhs, rhs.rightOperand() );
   }
   else if( !IsOperation_v<MT2> && isSame( ~lhs, rhs.rightOperand() ) ) {
      cudaAddAssign( ~lhs, rhs.leftOperand() );
   }
   else if( !RequiresEvaluation_v<MT2> ) {
      cudaAssign   ( ~lhs, rhs.rightOperand() );
      cudaAddAssign( ~lhs, rhs.leftOperand() );
   }
   else {
      cudaAssign   ( ~lhs, rhs.leftOperand() );
      cudaAddAssign( ~lhs, rhs.rightOperand() );
   }
}
   /*! \endcond */
   //**********************************************************************************************

} // namespace blaze

#endif