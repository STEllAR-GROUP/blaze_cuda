//=================================================================================================
/*!
//  \file blaze_cuda/math/cublas/dotu.h
//  \brief Header file for BLAS dot product (dotu)
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

#ifndef _BLAZE_CUDA_MATH_CUBLAS_DOTU_H_
#define _BLAZE_CUDA_MATH_CUBLAS_DOTU_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cublas.h>

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/NumericCast.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_cuda/math/CUDADynamicVector.h>


namespace blaze {

//=================================================================================================
//
//  BLAS WRAPPER FUNCTIONS (DOTU)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS wrapper functions (dotu) */
//@{
#if BLAZE_CUBLAS_MODE

BLAZE_ALWAYS_INLINE float cudotu( int n, const float* x, int incX, const float* y, int incY );

BLAZE_ALWAYS_INLINE double cudotu( int n, const double* x, int incX, const double* y, int incY );

BLAZE_ALWAYS_INLINE complex<float> cudotu( int n, const complex<float>* x, int incX,
                                           const complex<float>* y, int incY );

BLAZE_ALWAYS_INLINE complex<double> cudotu( int n, const complex<double>* x, int incX,
                                            const complex<double>* y, int incY );

template< typename VT1, bool TF1, typename VT2, bool TF2 >
BLAZE_ALWAYS_INLINE ElementType_t<VT1> dotu( const CUDADynamicVector<VT1,TF1>& x, const CUDADynamicVector<VT2,TF2>& y );

#endif
//@}
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a dense vector dot product for single precision operands
//        (\f$ s=\vec{x}*\vec{y} \f$).
// \ingroup blas
//
// \param n The size of the two dense vectors \a x and \a y \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense vector dot product for single precision operands based on
// the BLAS cublasSdot() function.
*/
BLAZE_ALWAYS_INLINE float cudotu( int n, const float* x, int incX, const float* y, int incY )
{
   cudaHandle_t handle;
   cublasCreate( &handle );
   auto ret = cublasSdot( &handle, n, x, incX, y, incY );
   cublasDestroy( handle );
   return ret;
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a dense vector dot product for double precision operands
//        (\f$ s=\vec{x}*\vec{y} \f$).
// \ingroup blas
//
// \param n The size of the two dense vectors \a x and \a y \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense vector dot product for double precision operands based on
// the BLAS cublasDdot() function.
*/
BLAZE_ALWAYS_INLINE double cudotu( int n, const double* x, int incX, const double* y, int incY )
{
   cudaHandle_t handle;
   cublasCreate( &handle );
   auto ret = cublasDdot( &handle, n, x, incX, y, incY );
   cublasDestroy( handle );
   return ret;
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a dense vector dot product for single precision complex operands
//        (\f$ s=\vec{x}*\vec{y} \f$).
// \ingroup blas
//
// \param n The size of the two dense vectors \a x and \a y \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense vector dot product for single precision complex operands
// based on the BLAS cublasCdotu_sub() function.
*/
BLAZE_ALWAYS_INLINE complex<float> cudotu( int n, const complex<float>* x, int incX,
                                           const complex<float>* y, int incY )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   complex<float> tmp;

#ifdef OPENBLAS_VERSION
   cublasCdotu_sub( n, reinterpret_cast<const float*>( x ), incX,
                    reinterpret_cast<const float*>( y ), incY,
                    reinterpret_cast<openblas_complex_float*>( &tmp ) );
#else
   cublasCdotu_sub( n, reinterpret_cast<const float*>( x ), incX,
                    reinterpret_cast<const float*>( y ), incY, &tmp );
#endif

   return tmp;
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a dense vector dot product for double precision complex operands
//        (\f$ s=\vec{x}*\vec{y} \f$).
// \ingroup blas
//
// \param n The size of the two dense vectors \a x and \a y \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense vector dot product for double precision complex operands
// based on the BLAS cublasZdotu_sub() function.
*/
BLAZE_ALWAYS_INLINE complex<double> cudotu( int n, const complex<double>* x, int incX,
                                            const complex<double>* y, int incY )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   complex<double> tmp;

   cublasZdotu_sub( n, reinterpret_cast<const double*>( x ), incX,
                    reinterpret_cast<const double*>( y ), incY, &tmp );

   return tmp;
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a dense vector dot product (\f$ s=\vec{x}*\vec{y} \f$).
// \ingroup blas
//
// \param x The left-hand side dense vector operand.
// \param y The right-hand side dense vector operand.
// \return void
//
// This function performs the dense vector dot product based on the BLAS dotu() functions. Note
// that the function only works for vectors with \c float, \c double, \c complex<float>, or
// \c complex<double> element type. The attempt to call the function with vectors of any other
// element type results in a compile time error.
*/
template< typename VT1, bool TF1, typename VT2, bool TF2 >
ElementType_t<VT1> dotu( const CUDADynamicVector<VT1,TF1>& x, const CUDADynamicVector<VT2,TF2>& y )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT2 );

   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS( VT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS( VT2 );

   BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT1> );
   BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT2> );

   const int n( numeric_cast<int>( (~x).size() ) );

   return cudotu( n, (~x).data(), 1, (~y).data(), 1 );
}
#endif
//*************************************************************************************************

} // namespace blaze

#endif
