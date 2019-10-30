//=================================================================================================
/*!
//  \file blaze_cuda/math/cuda/DenseVector.h
//  \brief Header file for the CUDA-based dense vector SMP implementation
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

#ifndef _BLAZE_CUDA_MATH_CUDA_DENSEVECTOR_H_
#define _BLAZE_CUDA_MATH_CUDA_DENSEVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/SMPAssignable.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/smp/SerialSection.h>
#include <blaze/math/smp/Functions.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsCUDAAssignable.h>
#include <blaze/math/views/Subvector.h>
#include <blaze/system/SMP.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/Assert.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/FunctionTrace.h>
#include <blaze/util/StaticAssert.h>
#include <blaze/util/Types.h>

#include <blaze_cuda/util/algorithms/CUDATransform.h>
#include <blaze_cuda/util/CUDAErrorManagement.h>


namespace blaze {

//=================================================================================================
//
//  CUDA-BASED ASSIGNMENT KERNELS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend of the CUDA-based (compound) assignment of a dense vector to a dense vector.
// \ingroup cuda
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side dense vector to be assigned.
// \param op The (compound) assignment operation.
// \return void
//
// This function is the backend implementation of the CUDA-based assignment of a dense
// vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1   // Type of the left-hand side dense vector
        , bool TF1       // Transpose flag of the left-hand side dense vector
        , typename VT2   // Type of the right-hand side dense vector
        , bool TF2       // Transpose flag of the right-hand side dense vector
        , typename OP >  // Type of the assignment operation
inline void cudaAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs, OP op )
{
   BLAZE_FUNCTION_TRACE;

   cuda_transform( (~lhs).begin(), (~lhs).end(), (~rhs).begin(), (~lhs).begin(), op );

   CUDA_ERROR_CHECK;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Backend of the CUDA-based (compound) assignment of a sparse vector to a dense vector.
// \ingroup cuda
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side sparse vector to be assigned.
// \param op The (compound) assignment operation.
// \return void
//
// This function is the backend implementation of the CUDA-based assignment of a sparse
// vector to a dense vector.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
//template< typename VT1   // Type of the left-hand side dense vector
//        , bool TF1       // Transpose flag of the left-hand side dense vector
//        , typename VT2   // Type of the right-hand side sparse vector
//        , bool TF2       // Transpose flag of the right-hand side sparse vector
//        , typename OP >  // Type of the assignment operation
//void cudaAssign( DenseVector<VT1,TF1>& lhs, const SparseVector<VT2,TF2>& rhs, OP op )
//{
//   (void)lhs; (void)rhs; (void)op;
//}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  PLAIN ASSIGNMENT
//
//=================================================================================================

template< typename VT1   // Type of the left-hand side dense vector
        , bool TF1       // Transpose flag of the left-hand side dense vector
        , typename VT2   // Type of the right-hand side dense vector
        , bool TF2 >     // Transpose flag of the right-hand side dense vector
inline auto cudaAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   cudaAssign( ~lhs, ~rhs, [] __device__ ( auto const&, auto const& r ) { return r; } );
}




//=================================================================================================
//
//  ADDITION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the CUDA-based addition assignment to a dense vector.
// \ingroup cuda
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side sparse vector to be added.
// \return void
//
// This function implements the CUDA-based addition assignment to a dense vector. Due to
// the explicit application of the SFINAE principle, this function can only be selected by the
// compiler in case both operands are SMP-assignable and the element types of both operands are
// not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto cudaAddAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).size() == (~rhs).size(), "Invalid vector sizes" );

   cudaAssign( ~lhs, ~rhs, [] __device__ ( auto const& l, auto const& r ) { return l + r; } );
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
/*!\brief Implementation of the CUDA-based subtraction assignment to a dense vector.
// \ingroup cuda
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side sparse vector to be subtracted.
// \return void
//
// This function implements the CUDA-based subtraction assignment to a dense vector. Due
// to the explicit application of the SFINAE principle, this function can only be selected by
// the compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto cudaSubAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).size() == (~rhs).size(), "Invalid vector sizes" );

   cudaAssign( ~lhs, ~rhs, [] __device__ ( auto const& l, auto const& r ) { return l - r; } );
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
/*!\brief Implementation of the CUDA-based multiplication assignment to a dense vector.
// \ingroup cuda
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side dense vector to be multiplied.
// \return void
//
// This function implements the CUDA-based multiplication assignment to a dense vector.
// Due to the explicit application of the SFINAE principle, this function can only be selected
// by the compiler in case both operands are SMP-assignable and the element types of both
// operands are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto cudaMultAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).size() == (~rhs).size(), "Invalid vector sizes" );

   cudaAssign( ~lhs, ~rhs, [] __device__ ( auto const& l, auto const& r ) { return l * r; } );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DIVISION ASSIGNMENT
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Implementation of the CUDA-based division assignment to a dense vector.
// \ingroup cuda
//
// \param lhs The target left-hand side dense vector.
// \param rhs The right-hand side dense vector divisor.
// \return void
//
// This function implements the CUDA-based division assignment to a dense vector. Due to
// the explicit application of the SFINAE principle, this function can only be selected by the
// compiler in case both operands are SMP-assignable and the element types of both operands
// are not SMP-assignable.\n
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto cudaDivAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
{
   BLAZE_FUNCTION_TRACE;

   BLAZE_INTERNAL_ASSERT( (~lhs).size() == (~rhs).size(), "Invalid vector sizes" );

   cudaAssign( ~lhs, ~rhs, [] __device__ ( auto const& l, auto const& r ) { return l / r; } );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  smpAssign() OVERLOADS
//
//=================================================================================================

template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1       // Transpose flag of the left-hand side dense vector
        , typename VT2   // Type of the right-hand side dense vector
        , bool TF2       // Transpose flag of the right-hand side dense vector
        , typename OP >  // Type of the assignment operation
inline auto smpAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<VT1> && IsCUDAAssignable_v<VT2> >
{
   BLAZE_FUNCTION_TRACE;

   cudaAssign( ~lhs, ~rhs );
}

template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpAddAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<VT1> && IsCUDAAssignable_v<VT2> >
{
   BLAZE_FUNCTION_TRACE;

   cudaAddAssign( ~lhs, ~rhs );
}

template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpSubAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<VT1> && IsCUDAAssignable_v<VT2> >
{
   BLAZE_FUNCTION_TRACE;

   cudaSubAssign( ~lhs, ~rhs );
}

template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpMultAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<VT1> && IsCUDAAssignable_v<VT2> >
{
   BLAZE_FUNCTION_TRACE;

   cudaMultAssign( ~lhs, ~rhs );
}

template< typename VT1  // Type of the left-hand side dense vector
        , bool TF1      // Transpose flag of the left-hand side dense vector
        , typename VT2  // Type of the right-hand side vector
        , bool TF2 >    // Transpose flag of the right-hand side vector
inline auto smpDivAssign( DenseVector<VT1,TF1>& lhs, const DenseVector<VT2,TF2>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<VT1> && IsCUDAAssignable_v<VT2> >
{
   BLAZE_FUNCTION_TRACE;

   cudaDivAssign( ~lhs, ~rhs );
}




//=================================================================================================
//
//  COMPILE TIME CONSTRAINTS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
namespace {

BLAZE_STATIC_ASSERT( BLAZE_CUDA_MODE );

}
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
