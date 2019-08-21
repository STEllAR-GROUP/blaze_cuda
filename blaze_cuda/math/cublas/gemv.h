//=================================================================================================
/*!
//  \file blaze_cuda/math/cublas/gemv.h
//  \brief Header file for BLAS general matrix/vector multiplication functions (gemv)
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

#ifndef _BLAZE_CUDA_MATH_CUBLAS_GEMV_H_
#define _BLAZE_CUDA_MATH_CUBLAS_GEMV_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cublas_v2.h>

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/NumericCast.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_cuda/math/DenseMatrix.h>
#include <blaze_cuda/math/DenseVector.h>


namespace blaze {

//=================================================================================================
//
//  BLAS WRAPPER FUNCTIONS (GEMV)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS wrapper functions (gemv) */
//@{
BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 float alpha, const float* A, int lda, const float* x, int incX,
                                 float beta, float* y, int incY );

BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 double alpha, const double* A, int lda, const double* x, int incX,
                                 double beta, double* y, int incY );

BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 complex<float> alpha, const complex<float>* A, int lda,
                                 const complex<float>* x, int incX, complex<float> beta,
                                 complex<float>* y, int incY );

BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 complex<double> alpha, const complex<double>* A, int lda,
                                 const complex<double>* x, int incX, complex<double> beta,
                                 complex<double>* y, int incY );

template< typename VT1, typename MT1, bool SO, typename VT2, typename ST >
BLAZE_ALWAYS_INLINE void cugemv(       DenseVector<VT1,blaze::columnMajor>& y,
                                 const DenseMatrix<MT1,SO>& A,
                                 const DenseVector<VT2,blaze::columnMajor>& x,
                                 ST alpha, ST beta );

template< typename VT1, typename VT2, typename MT1, bool SO, typename ST >
BLAZE_ALWAYS_INLINE void cugemv(       DenseVector<VT1,blaze::rowMajor>& y,
                                 const DenseVector<VT2,blaze::rowMajor>& x,
                                 const DenseMatrix<MT1,SO>& A,
                                 ST alpha, ST beta );
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense vector multiplication for single precision operands
//        (\f$ \vec{y}=\alpha*A*\vec{x}+\beta*\vec{y} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a A \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*\vec{x} \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param beta The scaling factor for \f$ \vec{y} \f$.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense matrix/dense vector multiplication for single precision
// operands based on the BLAS cublasSgemv() function.
*/
BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 float alpha, const float* A, int lda, const float* x, int incX,
                                 float beta, float* y, int incY )
{
   cublasHandle_t handle;
   cublasCreate_v2( &handle );
   cublasSgemv( handle, transA, m, n, &alpha, A, lda, x, incX, &beta, y, incY );
   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense vector multiplication for double precision operands
//        (\f$ \vec{y}=\alpha*A*\vec{x}+\beta*\vec{y} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a A \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*\vec{x} \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param beta The scaling factor for \f$ \vec{y} \f$.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense matrix/dense vector multiplication for double precision
// operands based on the BLAS cublasDgemv() function.
*/
BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 double alpha, const double* A, int lda, const double* x, int incX,
                                 double beta, double* y, int incY )
{
   cublasHandle_t handle;
   cublasCreate_v2( &handle );
   cublasDgemv( handle, transA, m, n, &alpha, A, lda, x, incX, &beta, y, incY );
   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense vector multiplication for single precision complex
//        operands (\f$ \vec{y}=\alpha*A*\vec{x}+\beta*\vec{y} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a A \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*\vec{x} \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param beta The scaling factor for \f$ \vec{y} \f$.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense matrix/dense vector multiplication for single precision
// complex operands based on the BLAS cublasCgemv() function.
*/
BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 complex<float> alpha, const complex<float>* A, int lda,
                                 const complex<float>* x, int incX, complex<float> beta,
                                 complex<float>* y, int incY )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cublasHandle_t handle;
   cublasCreate_v2( &handle );
   cublasCgemv( handle, transA, m, n,
      reinterpret_cast<const cuFloatComplex*>( &alpha ),
      reinterpret_cast<const cuFloatComplex*>( A ), lda,
      reinterpret_cast<const cuFloatComplex*>( x ), incX,
      reinterpret_cast<const cuFloatComplex*>( &beta ),
      reinterpret_cast<cuFloatComplex*>( y ), incY );
   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense vector multiplication for double precision complex
//        operands (\f$ \vec{y}=\alpha*A*\vec{x}+\beta*\vec{y} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a A \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*\vec{x} \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \param beta The scaling factor for \f$ \vec{y} \f$.
// \param y Pointer to the first element of vector \a y.
// \param incY The stride within vector \a y.
// \return void
//
// This function performs the dense matrix/dense vector multiplication for double precision
// complex operands based on the BLAS zblas_zgemv() function.
*/
BLAZE_ALWAYS_INLINE void cugemv( cublasOperation_t transA, int m, int n,
                                 complex<double> alpha, const complex<double>* A, int lda,
                                 const complex<double>* x, int incX, complex<double> beta,
                                 complex<double>* y, int incY )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   cublasHandle_t handle;
   cublasCreate_v2( &handle );
   cublasZgemv( handle, transA, m, n,
      reinterpret_cast<const cuDoubleComplex*>( &alpha ),
      reinterpret_cast<const cuDoubleComplex*>( A ), lda,
      reinterpret_cast<const cuDoubleComplex*>( x ), incX,
      reinterpret_cast<const cuDoubleComplex*>( &beta ),
      reinterpret_cast<cuDoubleComplex*>( y ), incY );
   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense vector multiplication
//        (\f$ \vec{y}=\alpha*A*\vec{x}+\beta*\vec{y} \f$).
// \ingroup blas
//
// \param y The target left-hand side dense vector.
// \param A The left-hand side dense matrix operand.
// \param x The right-hand side dense vector operand.
// \param alpha The scaling factor for \f$ A*\vec{x} \f$.
// \param beta The scaling factor for \f$ \vec{y} \f$.
// \return void
//
// This function performs the dense matrix/dense vector multiplication based on the BLAS cugemv()
// functions. Note that the function only works for vectors and matrices with \c float, \c double,
// \c complex<float>, or \c complex<double> element type. The attempt to call the function with
// vectors and matrices of any other element type results in a compile time error.
*/
template< typename VT1   // Type of the left-hand side target vector
        , typename MT1   // Type of the left-hand side matrix operand
        , bool SO        // Storage order of the left-hand side matrix operand
        , typename VT2   // Type of the right-hand side vector operand
        , typename ST >  // Type of the scalar factors
BLAZE_ALWAYS_INLINE void cugemv( DenseVector<VT1,false>& y,
                                 const DenseMatrix<MT1,SO>& A,
                                 const DenseVector<VT2,false>& x, ST alpha, ST beta )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT2 );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( VT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( VT2 );

   //BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT1> );
   //BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<MT1> );
   //BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT2> );

   const int m  ( numeric_cast<int>( SO == blaze::columnMajor ? (~A).rows() : (~A).columns() ) );
   const int n  ( numeric_cast<int>( SO == blaze::columnMajor ? (~A).columns() : (~A).rows() ) );
   const int lda( numeric_cast<int>( n ) );

   cugemv( SO == blaze::columnMajor ? CUBLAS_OP_N : CUBLAS_OP_T, m, n, alpha,
           (~A).data(), lda, (~x).data(), 1, beta, (~y).data(), 1 );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a transpose dense vector/dense matrix multiplication
//        (\f$ \vec{y}^T=\alpha*\vec{x}^T*A+\beta*\vec{y}^T \f$).
// \ingroup blas
//
// \param y The target left-hand side dense vector.
// \param x The left-hand side dense vector operand.
// \param A The right-hand side dense matrix operand.
// \param alpha The scaling factor for \f$ \vec{x}^T*A \f$.
// \param beta The scaling factor for \f$ \vec{y}^T \f$.
// \return void
//
// This function performs the transpose dense vector/dense matrix multiplication based on the
// BLAS cugemv() functions. Note that the function only works for vectors and matrices with \c float,
// \c double, \c complex<float>, or \c complex<double> element type. The attempt to call the
// function with vectors and matrices of any other element type results in a compile time error.
*/
template< typename VT1   // Type of the left-hand side target vector
        , typename VT2   // Type of the left-hand side vector operand
        , typename MT1   // Type of the right-hand side matrix operand
        , bool SO        // Storage order of the right-hand side matrix operand
        , typename ST >  // Type of the scalar factors
BLAZE_ALWAYS_INLINE void cugemv( DenseVector<VT1,true>& y,
                                 const DenseVector<VT2,true>& x,
                                 const DenseMatrix<MT1,SO>& A, ST alpha, ST beta )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT2 );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( VT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( VT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT1 );

   //BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT1> );
   //BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<MT1> );
   //BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT2> );

   const int m  ( numeric_cast<int>( SO == blaze::columnMajor ? (~A).rows() : (~A).columns() ) );
   const int n  ( numeric_cast<int>( SO == blaze::columnMajor ? (~A).columns() : (~A).rows() ) );
   const int lda( numeric_cast<int>( m ) );

   cugemv( SO == blaze::columnMajor ? CUBLAS_OP_N : CUBLAS_OP_T, m, n, alpha,
           (~A).data(), lda, (~x).data(), 1, beta, (~y).data(), 1 );
}
//*************************************************************************************************

} // namespace blaze

#endif
