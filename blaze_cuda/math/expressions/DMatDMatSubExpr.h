//=================================================================================================
/*!
//  \file blaze_cuda/math/expressions/DMatDMatSubExpr.h
//  \brief Header file for the dense matrix/dense matrix multiplication expression
//
//  Copyright (C) 2019 Jules Penuchot - All Rights Reserved
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

#ifndef _BLAZE_CUDA_MATH_EXPRESSIONS_DMATDMATSUBEXPR_H_
#define _BLAZE_CUDA_MATH_EXPRESSIONS_DMATDMATSUBEXPR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/expressions/DMatDMatSubExpr.h>
#include <blaze/math/traits/DeclSymTrait.h>

#include <blaze_cuda/math/typetraits/RequiresCUDAEvaluation.h>


namespace blaze {

//**Assignment to dense matrices****************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment of a dense matrix-dense matrix addition to a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side addition expression to be assigned.
// \return auto
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
inline auto cudaAssign( DenseMatrix<MT,SO2>& lhs, const DMatDMatSubExpr<MT1,MT2,SO>& rhs )
   -> EnableIf_t< RequiresCUDAEvaluation_v<MT1> || RequiresCUDAEvaluation_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   // TODO
}
/*! \endcond */
//**********************************************************************************************

//**Addition assignment to dense matrices*******************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment of a dense matrix-dense matrix addition to a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side addition expression to be added.
// \return auto
//
// This function implements the performance optimized addition assignment of a dense matrix-
// dense matrix addition expression to a dense matrix. Due to the explicit application of
// the SFINAE principle, this function can only be selected by the compiler in case either
// of the operands requires an intermediate evaluation.
*/
template< typename MT  // Type of the target dense matrix
            , bool SO2     // Storage order of the target dense matrix
            , typename MT1
            , typename MT2
            , bool SO >
inline auto cudaAddAssign( DenseMatrix<MT,SO2>& lhs, const DMatDMatSubExpr<MT1,MT2,SO>& rhs )
    -> EnableIf_t< RequiresCUDAEvaluation_v<MT1> || RequiresCUDAEvaluation_v<MT2> >
{
    BLAZE_FUNCTION_TRACE;

    // TODO
}
/*! \endcond */
//**********************************************************************************************

//**Subtraction assignment to dense matrices****************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment of a dense matrix-dense matrix addition to a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side addition expression to be subtracted.
// \return void
//
// This function implements the performance optimized subtraction assignment of a dense matrix-
// dense matrix addition expression to a dense matrix. Due to the explicit application of
// the SFINAE principle, this function can only be selected by the compiler in case either
// of the operands requires an intermediate evaluation.
*/
template< typename MT  // Type of the target dense matrix
            , bool SO2     // Storage order of the target dense matrix
            , typename MT1
            , typename MT2
            , bool SO >
inline auto cudaSubAssign( DenseMatrix<MT,SO2>& lhs, const DMatDMatSubExpr<MT1,MT2,SO>& rhs )
    -> EnableIf_t< RequiresCUDAEvaluation_v<MT1> || RequiresCUDAEvaluation_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   // TODO
}
/*! \endcond */
//**********************************************************************************************

//**Schur product assignment to dense matrices**************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment of a dense matrix-dense matrix addition to a dense matrix.
// \ingroup dense_matrix
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side addition expression for the Schur product.
// \return void
//
// This function implements the performance optimized Schur product assignment of a dense
// matrix-dense matrix addition expression to a dense matrix. Due to the explicit application
// of the SFINAE principle, this function can only be selected by the compiler in case either
// of the operands requires an intermediate evaluation.
*/
template< typename MT  // Type of the target dense matrix
            , bool SO2     // Storage order of the target dense matrix
            , typename MT1
            , typename MT2
            , bool SO >
inline auto cudaSchurAssign( DenseMatrix<MT,SO2>& lhs, const DMatDMatSubExpr<MT1,MT2,SO>& rhs )
    -> EnableIf_t< RequiresCUDAEvaluation_v<MT1> || RequiresCUDAEvaluation_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   // TODO
}
/*! \endcond */
//**********************************************************************************************

template< typename MT1, typename MT2, bool SO >
struct RequiresCUDAEvaluation< DMatDMatSubExpr<MT1,MT2,SO>
   , EnableIf_t< IsCUDAAssignable_v< DMatDMatSubExpr<MT1,MT2,SO> > > >
{
public:
   static constexpr bool value = RequiresCUDAEvaluation_v<MT1> || RequiresCUDAEvaluation_v<MT2>;
};

} // namespace blaze

#endif
