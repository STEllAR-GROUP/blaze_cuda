//=================================================================================================
/*!
//  \file blaze/math/cuda/DenseMatrix.h
//  \brief Header file for the CUDA-based dense matrix SMP implementation
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

#ifndef _BLAZE_MATH_CUDA_DENSEMATRIX_H_
#define _BLAZE_MATH_CUDA_DENSEMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/SparseMatrix.h>
//#include <blaze/math/functors/SchurAssign.h>
#include <blaze/math/smp/SerialSection.h>
#include <blaze/math/StorageOrder.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze/system/SMP.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>

#include <blaze_cuda/math/typetraits/IsCUDAAssignable.h>
#include <blaze_cuda/util/algorithms/CUDATransform.h>

namespace blaze {

//=================================================================================================
//
//  CUDA-BASED ASSIGNMENT KERNELS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend of the CUDA-based (compound) assignment of a dense matrix to a dense matrix.
// \ingroup math
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side dense matrix to be assigned.
// \param op The (compound) assignment operation.
// \return void
//
// This function is the backend implementation of the CUDA-based assignment of a dense
// matrix to a dense matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1   // Type of the left-hand side dense matrix
        , bool SO1       // Storage order of the left-hand side dense matrix
        , typename MT2   // Type of the right-hand side dense matrix
        , bool SO2       // Storage order of the right-hand side dense matrix
        , typename OP >  // Type of the assignment operation
void cudaAssign( DenseMatrix<MT1,SO1>& lhs, const DenseMatrix<MT2,SO2>& rhs, OP op )
{
   // TODO
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend of the CUDA-based (compound) assignment of a sparse matrix to a dense matrix.
// \ingroup math
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side sparse matrix to be assigned.
// \param op The (compound) assignment operation.
// \return void
//
// This function is the backend implementation of the CUDA-based assignment of a sparse
// matrix to a dense matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1   // Type of the left-hand side dense matrix
        , bool SO1       // Storage order of the left-hand side dense matrix
        , typename MT2   // Type of the right-hand side sparse matrix
        , bool SO2       // Storage order of the right-hand side sparse matrix
        , typename OP >  // Type of the assignment operation
void cudaAssign( DenseMatrix<MT1,SO1>& lhs, const SparseMatrix<MT2,SO2>& rhs, OP op )
{
   // TODO
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  PLAIN ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the CUDA-based assignment to a dense matrix.
// \ingroup smp
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
//
// This function implements the default CUDA-based assignment to a dense matrix. Due to
// the explicit application of the SFINAE principle, this function can only be selected by the
// compiler in case both operands are SMP-assignable and the element types of both operands are
// not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto cudaAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && ( !IsCUDAAssignable_v<MT1> || !IsCUDAAssignable_v<MT2> ) >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   assign( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the CUDA-based assignment to a dense matrix.
// \ingroup math
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be assigned.
// \return void
//
// This function implements the CUDA-based assignment to a dense matrix. Due to the
// explicit application of the SFINAE principle, this function can only be selected by the
// compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && IsCUDAAssignable_v<MT1> && IsCUDAAssignable_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   cudaAssign( ~lhs, ~rhs, Assign() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDITION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the CUDA-based addition assignment to a dense matrix.
// \ingroup smp
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
//
// This function implements the default CUDA-based addition assignment to a dense matrix.
// Due to the explicit application of the SFINAE principle, this function can only be selected
// by the compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpAddAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && ( !IsCUDAAssignable_v<MT1> || !IsCUDAAssignable_v<MT2> ) >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   addAssign( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the CUDA-based addition assignment to a dense matrix.
// \ingroup math
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be added.
// \return void
//
// This function implements the CUDA-based addition assignment to a dense matrix. Due to
// the explicit application of the SFINAE principle, this function can only be selected by the
// compiler in case both operands are SMP-assignable and the element types of both operands are
// not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpAddAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && IsCUDAAssignable_v<MT1> && IsCUDAAssignable_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   cudaAssign( ~lhs, ~rhs, AddAssign() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRACTION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the CUDA-based subtracction assignment to a dense matrix.
// \ingroup smp
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
//
// This function implements the default CUDA-based subtraction assignment to a dense matrix.
// Due to the explicit application of the SFINAE principle, this function can only be selected by
// the compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpSubAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && ( !IsCUDAAssignable_v<MT1> || !IsCUDAAssignable_v<MT2> ) >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   subAssign( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the CUDA-based subtracction assignment to a dense matrix.
// \ingroup smp
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be subtracted.
// \return void
//
// This function implements the default CUDA-based subtraction assignment of a matrix to a
// dense matrix. Due to the explicit application of the SFINAE principle, this function can only
// be selected by the compiler in case both operands are SMP-assignable and the element types of
// both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpSubAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && IsCUDAAssignable_v<MT1> && IsCUDAAssignable_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   cudaAssign( ~lhs, ~rhs, SubAssign() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SCHUR PRODUCT ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the CUDA-based Schur product assignment to a dense matrix.
// \ingroup smp
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix for the Schur product.
// \return void
//
// This function implements the default CUDA-based Schur product assignment to a dense
// matrix. Due to the explicit application of the SFINAE principle, this function can only be
// selected by the compiler in case both operands are SMP-assignable and the element types of
// both operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpSchurAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && ( !IsCUDAAssignable_v<MT1> || !IsCUDAAssignable_v<MT2> ) >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   schurAssign( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the CUDA-based Schur product assignment to a dense matrix.
// \ingroup math
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix for the Schur product.
// \return void
//
// This function implements the CUDA-based Schur product assignment to a dense matrix. Due
// to the explicit application of the SFINAE principle, this function can only be selected by the
// compiler in case both operands are SMP-assignable and the element types of both operands are
// not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side dense matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpSchurAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> && IsCUDAAssignable_v<MT1> && IsCUDAAssignable_v<MT2> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_NOT_BE_SMP_ASSIGNABLE( ElementType_t<MT2> );

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   cudaAssign( ~lhs, ~rhs, SchurAssign() );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTIPLICATION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the CUDA-based multiplication assignment to a dense matrix.
// \ingroup smp
//
// \param lhs The target left-hand side dense matrix.
// \param rhs The right-hand side matrix to be multiplied.
// \return void
//
// This function implements the default CUDA-based multiplication assignment to a dense
// matrix.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename MT1  // Type of the left-hand side dense matrix
        , bool SO1      // Storage order of the left-hand side matrix
        , typename MT2  // Type of the right-hand side matrix
        , bool SO2 >    // Storage order of the right-hand side matrix
inline auto smpMultAssign( Matrix<MT1,SO1>& lhs, const Matrix<MT2,SO2>& rhs )
   -> EnableIf_t< IsDenseMatrix_v<MT1> >
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).rows()    == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( (~lhs).columns() == (~rhs).columns(), "Invalid number of columns" );

   multAssign( ~lhs, ~rhs );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COMPILE TIME CONSTRAINT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
//namespace {
//
//BLAZE_STATIC_ASSERT( BLAZE_HPX_PARALLEL_MODE );
//
//}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
