//=================================================================================================
/*!
//  \file blaze_cuda/math/cublas/trmv.h
//  \brief Header file for BLAS triangular matrix/vector multiplication functions (trmv)
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

#ifndef _BLAZE_CUDA_MATH_CUBLAS_TRMV_H_
#define _BLAZE_CUDA_MATH_CUBLAS_TRMV_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cublas.h>

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/NumericCast.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_cuda/math/CUDADynamicMatrix.h>
#include <blaze_cuda/math/CUDADynamicVector.h>


namespace blaze {

//=================================================================================================
//
//  BLAS WRAPPER FUNCTIONS (TRMV)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS wrapper functions (trmv) */
//@{
#if BLAZE_CUBLAS_MODE

BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const float* A, int lda, float* x,
                                 int incX );

BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const double* A, int lda, double* x,
                                 int incX );

BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const complex<float>* A, int lda,
                                 complex<float>* x, int incX );

BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const complex<double>* A, int lda,
                                 complex<double>* x, int incX );

template< typename VT, typename MT, bool SO >
BLAZE_ALWAYS_INLINE void trmv( CUDADynamicVector<VT,false>& x, const CUDADynamicMatrix<MT,SO>& A,
                               CBLAS_UPLO uplo );

template< typename VT, typename MT, bool SO >
BLAZE_ALWAYS_INLINE void trmv( CUDADynamicVector<VT,true>& x, const CUDADynamicMatrix<MT,SO>& A,
                               CBLAS_UPLO uplo );

#endif
//@}
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense vector multiplication for single
//        precision operands (\f$ \vec{x}=A*\vec{x} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \return void
//
// This function performs the multiplication of a single precision triangular matrix by a vector
// based on the cublasStrmv() function.
*/
BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const float* A, int lda, float* x,
                                 int incX )
{
   cublasHandle_t handle;
   cublasCreate( &handle );
   cublasStrmv( handle, order, uplo, transA, diag, n, A, lda, x, incX );
   cublasDestroy( handle );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense vector multiplication for double
//        precision operands (\f$ \vec{x}=A*\vec{x} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \return void
//
// This function performs the multiplication of a double precision triangular matrix by a vector
// based on the cublasDtrmv() function.
*/
BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const double* A, int lda, double* x,
                                 int incX )
{
   cublasHandle_t handle;
   cublasCreate( &handle );
   cublasDtrmv( handle, order, uplo, transA, diag, n, A, lda, x, incX );
   cublasDestroy( handle );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense vector multiplication for single
//        precision complex operands (\f$ \vec{x}=A*\vec{x} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \return void
//
// This function performs the multiplication of a single precision complex triangular matrix by a
// vector based on the cublasCtrmv() function.
*/
BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const complex<float>* A, int lda,
                                 complex<float>* x, int incX )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cublasHandle_t handle;
   cublasCreate( &handle );
   cublasCtrmv( handle, order, uplo, transA, diag, n, reinterpret_cast<const float*>( A ),
                lda, reinterpret_cast<float*>( x ), incX );
   cublasDestroy( handle );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense vector multiplication for double
//        precision complex operands (\f$ \vec{x}=A*\vec{x} \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param diag Specifies whether \a A is unitriangular (\a CblasNonUnit or \a CblasUnit).
// \param n The number of rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param x Pointer to the first element of vector \a x.
// \param incX The stride within vector \a x.
// \return void
//
// This function performs the multiplication of a double precision complex triangular matrix by a
// vector based on the cublasZtrmv() function.
*/
BLAZE_ALWAYS_INLINE void cutrmv( CBLAS_ORDER order, CBLAS_UPLO uplo, CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag, int n, const complex<double>* A, int lda,
                                 complex<double>* x, int incX )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   cublasHandle_t handle;
   cublasCreate( &handle );
   cublasZtrmv( handle, order, uplo, transA, diag, n, reinterpret_cast<const double*>( A ),
                lda, reinterpret_cast<double*>( x ), incX );
   cublasDestroy( handle );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a triangular dense matrix/dense vector multiplication
//        (\f$ \vec{x}=A*\vec{x} \f$).
// \ingroup blas
//
// \param x The target left-hand side dense vector.
// \param A The dense matrix operand.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \return void
//
// This function performs the multiplication of a triangular matrix by a vector based on the BLAS
// trmv() functions. Note that the function only works for vectors and matrices with \c float,
// \c double, \c complex<float>, or \c complex<double> element type. The attempt to call the
// function with vectors and matrices of any other element type results in a compile time error.
*/
template< typename VT  // Type of the target vector
        , typename MT  // Type of the matrix operand
        , bool SO >    // Storage order of the matrix operand
BLAZE_ALWAYS_INLINE void trmv( CUDADynamicVector<VT,false>& x, const CUDADynamicMatrix<MT,SO>& A,
                               CBLAS_UPLO uplo )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( VT );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT );

   BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT> );
   BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<MT> );

   BLAZE_INTERNAL_ASSERT( (~A).rows() == (~A).columns(), "Non-square triangular matrix detected" );
   BLAZE_INTERNAL_ASSERT( uplo == CblasLower || uplo == CblasUpper, "Invalid uplo argument detected" );

   const int n  ( numeric_cast<int>( (~A).rows() )    );
   const int lda( numeric_cast<int>( (~A).spacing() ) );

   cutrmv( ( IsRowMajorMatrix_v<MT> )?( CblasRowMajor ):( CblasColMajor ),
           uplo, CblasNoTrans, CblasNonUnit, n, (~A).data(), lda, (~x).data(), 1 );
}
#endif
//*************************************************************************************************


//*************************************************************************************************
#if BLAZE_CUBLAS_MODE
/*!\brief BLAS kernel for a transpose dense vector/triangular dense matrix multiplication
//        (\f$ \vec{x}^T=\vec{x}^T*A \f$).
// \ingroup blas
//
// \param x The target left-hand side dense vector.
// \param A The dense matrix operand.
// \param uplo \a CblasLower to use the lower triangle from \a A, \a CblasUpper to use the upper triangle.
// \return void
//
// This function performs the multiplication of a vector and a triangular matrix based on the BLAS
// trmv() functions. Note that the function only works for vectors and matrices with \c float,
// \c double, \c complex<float>, or \c complex<double> element type. The attempt to call the
// function with vectors and matrices of any other element type results in a compile time error.
*/
template< typename VT  // Type of the target vector
        , typename MT  // Type of the matrix operand
        , bool SO >    // Storage order of the matrix operand
BLAZE_ALWAYS_INLINE void trmv( CUDADynamicVector<VT,true>& x, const CUDADynamicMatrix<MT,SO>& A,
                               CBLAS_UPLO uplo )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( VT );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( VT );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT );

   BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<VT> );
   BLAZE_CONSTRAINT_MUST_BE_CUBLAS_COMPATIBLE_TYPE( ElementType_t<MT> );

   BLAZE_INTERNAL_ASSERT( (~A).rows() == (~A).columns(), "Non-square triangular matrix detected" );
   BLAZE_INTERNAL_ASSERT( uplo == CblasLower || uplo == CblasUpper, "Invalid uplo argument detected" );

   const int n  ( numeric_cast<int>( (~A).rows() )    );
   const int lda( numeric_cast<int>( (~A).spacing() ) );

   cutrmv( ( IsRowMajorMatrix_v<MT> )?( CblasRowMajor ):( CblasColMajor ),
           uplo, CblasTrans, CblasNonUnit, n, (~A).data(), lda, (~x).data(), 1 );
}
#endif
//*************************************************************************************************

} // namespace blaze

#endif
