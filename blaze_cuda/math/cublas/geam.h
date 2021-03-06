//=================================================================================================
/*!
//  \file blaze_cuda/math/cublas/geam.h
//  \brief Header file for BLAS general matrix/matrix multiplication functions (geam)
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

#ifndef _BLAZE_CUDA_MATH_CUBLAS_GEAM_H_
#define _BLAZE_CUDA_MATH_CUBLAS_GEAM_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cublas_v2.h>

#include <blaze/math/Aliases.h>
#include <blaze/math/constraints/BLASCompatible.h>
#include <blaze/math/constraints/Computation.h>
#include <blaze/math/constraints/ConstDataAccess.h>
#include <blaze/math/constraints/MutableDataAccess.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/system/BLAS.h>
#include <blaze/system/Inline.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Complex.h>
#include <blaze/util/NumericCast.h>
#include <blaze/util/StaticAssert.h>

#include <blaze_cuda/util/CUDAManagedAllocator.h>
#include <blaze_cuda/util/CUBLASErrorManagement.h>


namespace blaze {

//=================================================================================================
//
//  BLAS WRAPPER FUNCTIONS (GEMM)
//
//=================================================================================================

//*************************************************************************************************
/*!\name BLAS wrapper functions (geam) */
//@{

BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                 const float alpha, const float *A, int lda,
                                 const float beta , const float *B, int ldb,
                                                          float *C, int ldc );

BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                 const double alpha, const double *A, int lda,
                                 const double beta , const double *B, int ldb,
                                                           double *C, int ldc );

BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                 const complex<float> alpha, const complex<float> *A, int lda,
                                 const complex<float> beta , const complex<float> *B, int ldb,
                                                                   complex<float> *C, int ldc );

BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                 const complex<double> alpha, const complex<double> *A, int lda,
                                 const complex<double> beta , const complex<double> *B, int ldb,
                                                                    complex<double> *C, int ldc );

template< typename MT1, bool SO1, typename MT2, bool SO2, typename MT3, bool SO3, typename ST >
BLAZE_ALWAYS_INLINE void cugeam( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A,
                                 const DenseMatrix<MT3,SO3>& B, ST alpha, ST beta );

template< typename MT1, bool SO1, typename MT2, bool SO2, typename ST >
BLAZE_ALWAYS_INLINE void cugeam( DenseMatrix<MT1,SO1>& C, const DenseMatrix<MT2,SO2>& A, ST alpha );

//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with single precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for single precision
// matrices based on the BLAS cblas_sgeam() function.
*/
BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n,
                                 const float alpha, const float *A, int lda,
                                 const float beta , const float *B, int ldb,
                                                          float *C, int ldc )
{
   cublasHandle_t handle;
   cublasCreate_v2( &handle );

   // NB: Parameter numbering starts from handle = 0
   auto status = cublasSgeam( handle, transa, transb, m, n,
      &alpha, A, lda,
      &beta , B, ldb,
              C, ldc );

   CUBLAS_ERROR_CHECK( status );

   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with double precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for double precision
// matrices based on the BLAS cblas_dgeam() function.
*/
BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n,
                                 const double alpha, const double *A, int lda,
                                 const double beta , const double *B, int ldb,
                                                           double *C, int ldc )
{
   cublasHandle_t handle;
   cublasCreate_v2( &handle );

   // NB: Parameter numbering starts from handle = 0
   auto status = cublasDgeam( handle, transa, transb, m, n,
      &alpha, A, lda,
      &beta , B, ldb,
              C, ldc );

   CUBLAS_ERROR_CHECK( status );

   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with single precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for single precision
// complex matrices based on the BLAS cblas_cgeam() function.
*/
BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n,
                                 const complex<float> alpha, const complex<float> *A, int lda,
                                 const complex<float> beta , const complex<float> *B, int ldb,
                                                                   complex<float> *C, int ldc )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<float> ) == 2UL*sizeof( float ) );

   cublasHandle_t handle;
   cublasCreate_v2( &handle );

   // NB: Parameter numbering starts from handle = 0
   auto status = cublasCgeam( handle, transa, transb, m, n,
      reinterpret_cast<const cuFloatComplex*>( &alpha ),
      reinterpret_cast<const cuFloatComplex*>( A ), lda,
      reinterpret_cast<const cuFloatComplex*>( &beta ),
      reinterpret_cast<const cuFloatComplex*>( B ), ldb,
      reinterpret_cast<      cuFloatComplex*>( C ), ldc );

   CUBLAS_ERROR_CHECK( status );

   cublasDestroy_v2( handle );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication with double precision
//        matrices (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param order Specifies the storage order of matrix \a A (\a CblasColMajor or \a CblasColMajor).
// \param transA Specifies whether to transpose matrix \a A (\a CblasNoTrans or \a CblasTrans).
// \param transB Specifies whether to transpose matrix \a B (\a CblasNoTrans or \a CblasTrans).
// \param m The number of rows of matrix \a A and \a C \f$[0..\infty)\f$.
// \param n The number of columns of matrix \a B and \a C \f$[0..\infty)\f$.
// \param k The number of columns of matrix \a A and rows in matrix \a B \f$[0..\infty)\f$.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param A Pointer to the first element of matrix \a A.
// \param lda The total number of elements between two rows/columns of matrix \a A \f$[0..\infty)\f$.
// \param B Pointer to the first element of matrix \a B.
// \param ldb The total number of elements between two rows/columns of matrix \a B \f$[0..\infty)\f$.
// \param beta The scaling factor for \f$ C \f$.
// \param C Pointer to the first element of matrix \a C.
// \param ldc The total number of elements between two rows/columns of matrix \a C \f$[0..\infty)\f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication for double precision
// complex matrices based on the BLAS cblas_zgeam() function.
*/
BLAZE_ALWAYS_INLINE void cugeam( cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n,
                                 const complex<double> alpha, const complex<double> *A, int lda,
                                 const complex<double> beta , const complex<double> *B, int ldb,
                                                                    complex<double> *C, int ldc )
{
   BLAZE_STATIC_ASSERT( sizeof( complex<double> ) == 2UL*sizeof( double ) );

   cublasHandle_t handle;
   cublasCreate_v2( &handle );

   // NB: Parameter numbering starts from handle = 0
   auto status = cublasZgeam( handle, transa, transb, m, n,
      reinterpret_cast<const cuDoubleComplex*>( &alpha ),
      reinterpret_cast<const cuDoubleComplex*>( A ), lda,
      reinterpret_cast<const cuDoubleComplex*>( &beta ),
      reinterpret_cast<const cuDoubleComplex*>( B ), ldb,
      reinterpret_cast<      cuDoubleComplex*>( C ), ldc );

   CUBLAS_ERROR_CHECK( status );

   cublasDestroy_v2( handle );
}
//*************************************************************************************************


cublasOperation_t invertCublasOperation( cublasOperation_t const& op ) {
   if ( op == CUBLAS_OP_T ) return CUBLAS_OP_N;
   if ( op == CUBLAS_OP_N ) return CUBLAS_OP_T;
   return CUBLAS_OP_C;
}


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param C The target left-hand side dense matrix.
// \param A The left-hand side multiplication operand.
// \param B The right-hand side multiplication operand.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param beta The scaling factor for \f$ C \f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication based on the BLAS
// geam() functions. Note that the function only works for matrices with \c float, \c double,
// \c complex<float>, and \c complex<double> element type. The attempt to call the function
// with matrices of any other element type results in a compile time error.
*/
template< typename MT1   // Type of the left-hand side target matrix
        , bool SO1       // Storage order of the left-hand side target matrix
        , typename MT2   // Type of the left-hand side matrix operand
        , bool SO2       // Storage order of the left-hand side matrix operand
        , typename MT3   // Type of the right-hand side matrix operand
        , bool SO3       // Storage order of the right-hand side matrix operand
        , typename ST >  // Type of the scalar factors
BLAZE_ALWAYS_INLINE void cugeam ( DenseMatrix<MT1,SO1>& C,
                            const DenseMatrix<MT2,SO2>& A,
                            const DenseMatrix<MT3,SO3>& B,
                            ST alpha, cublasOperation_t transa ,
                            ST beta , cublasOperation_t transb )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT3 );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT2 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT3 );

   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT2> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT3> );

   const int m  ( numeric_cast<int>( SO1 == blaze::columnMajor ? (~C).rows() : (~C).columns() ) );
   const int n  ( numeric_cast<int>( SO1 == blaze::columnMajor ? (~C).columns() : (~C).rows() ) );
   const int lda( numeric_cast<int>( SO2 == blaze::columnMajor ? (~A).rows() : (~A).columns() ) );
   const int ldb( numeric_cast<int>( SO3 == blaze::columnMajor ? (~B).rows() : (~B).columns() ) );
   const int ldc( numeric_cast<int>( m ) );

   const cublasOperation_t ta( SO1 == SO2 ? transa : invertCublasOperation( transa ) );
   const cublasOperation_t tb( SO1 == SO2 ? transb : invertCublasOperation( transb ) );

   cugeam( ta, tb, m, n,
      alpha, (~A).data(), lda,
      beta,  (~B).data(), ldb,
             (~C).data(), ldc );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief BLAS kernel for a dense matrix/dense matrix multiplication (\f$ C=\alpha*A*B+\beta*C \f$).
// \ingroup blas
//
// \param C The target left-hand side dense matrix.
// \param A The left-hand side multiplication operand.
// \param B The right-hand side multiplication operand.
// \param alpha The scaling factor for \f$ A*B \f$.
// \param beta The scaling factor for \f$ C \f$.
// \return void
//
// This function performs the dense matrix/dense matrix multiplication based on the BLAS
// geam() functions. Note that the function only works for matrices with \c float, \c double,
// \c complex<float>, and \c complex<double> element type. The attempt to call the function
// with matrices of any other element type results in a compile time error.
*/
template< typename MT1   // Type of the left-hand side target matrix
        , bool SO1       // Storage order of the left-hand side target matrix
        , typename MT2   // Type of the left-hand side matrix operand
        , bool SO2       // Storage order of the left-hand side matrix operand
        , typename ST >  // Type of the scalar factors
BLAZE_ALWAYS_INLINE void cugeam ( DenseMatrix<MT1,SO1>& C,
                            const DenseMatrix<MT2,SO2>& A,
                            ST alpha, cublasOperation_t transa )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT1 );
   BLAZE_CONSTRAINT_MUST_NOT_BE_COMPUTATION_TYPE( MT2 );

   BLAZE_CONSTRAINT_MUST_HAVE_MUTABLE_DATA_ACCESS( MT1 );
   BLAZE_CONSTRAINT_MUST_HAVE_CONST_DATA_ACCESS  ( MT2 );

   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT1> );
   BLAZE_CONSTRAINT_MUST_BE_BLAS_COMPATIBLE_TYPE( ElementType_t<MT2> );

   const int m  ( numeric_cast<int>( SO1 == blaze::columnMajor ? (~C).rows() : (~C).columns() ) );
   const int n  ( numeric_cast<int>( SO1 == blaze::columnMajor ? (~C).columns() : (~C).rows() ) );
   const int lda( numeric_cast<int>( SO2 == blaze::columnMajor ? (~A).rows() : (~A).columns() ) );
   const int ldc( numeric_cast<int>( m ) );

   const cublasOperation_t ta ( SO1 == SO2 ? transa : invertCublasOperation(transa) );

   cugeam( ta, ta, m, n,
      alpha, (~A).data(), lda ,
      ST(0), (~A).data(), lda ,
             (~C).data(), ldc );
}
//*************************************************************************************************

} // namespace blaze

#endif
