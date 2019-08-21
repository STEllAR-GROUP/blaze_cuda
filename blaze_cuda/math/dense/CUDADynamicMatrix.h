//=================================================================================================
/*!
//  \file blaze_cuda/math/dense/CUDADynamicMatrix.h
//  \brief Header file for the implementation of a dynamic MxN matrix
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

#ifndef _BLAZE_CUDA_MATH_DENSE_CUDADYNAMICMATRIX_H_
#define _BLAZE_CUDA_MATH_DENSE_CUDADYNAMICMATRIX_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/Diagonal.h>
#include <blaze/math/constraints/Symmetric.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/SparseMatrix.h>
#include <blaze/math/Forward.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/Conjugate.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/shims/Reset.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/ColumnsTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ExpandTrait.h>
#include <blaze/math/traits/MapTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/RowsTrait.h>
#include <blaze/math/traits/SchurTrait.h>
#include <blaze/math/traits/SubmatrixTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HasSIMDAdd.h>
#include <blaze/math/typetraits/HasSIMDMult.h>
#include <blaze/math/typetraits/HasSIMDSub.h>
#include <blaze/math/typetraits/HighType.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsCUDAAssignable.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsDiagonal.h>
#include <blaze/math/typetraits/IsLower.h>
#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsShrinkable.h>
#include <blaze/math/typetraits/IsSIMDCombinable.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/math/typetraits/IsStrictlyLower.h>
#include <blaze/math/typetraits/IsStrictlyUpper.h>
#include <blaze/math/typetraits/IsSymmetric.h>
#include <blaze/math/typetraits/IsUpper.h>
#include <blaze/math/typetraits/LowType.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/typetraits/StorageOrder.h>
#include <blaze/system/Blocking.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Restrict.h>
#include <blaze/system/StorageOrder.h>
#include <blaze/system/Thresholds.h>
#include <blaze/system/TransposeFlag.h>
#include <blaze/util/algorithms/Max.h>
#include <blaze/util/algorithms/Min.h>
#include <blaze/util/algorithms/Transfer.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/constraints/Const.h>
#include <blaze/util/constraints/Pointer.h>
#include <blaze/util/constraints/Reference.h>
#include <blaze/util/constraints/Vectorizable.h>
#include <blaze/util/constraints/Volatile.h>
#include <blaze/util/DisableIf.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/MaybeUnused.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSame.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/typetraits/RemoveConst.h>

#include <blaze_cuda/util/Memory.h>
#include <blaze_cuda/util/algorithms/CUDATransform.h>
#include <blaze_cuda/util/CUDAErrorManagement.h>
#include <blaze_cuda/math/cuda/DenseMatrix.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dynamic_matrix CUDADynamicMatrix
// \ingroup dense_matrix
*/
/*!\brief Efficient implementation of a dynamic \f$ M \times N \f$ matrix.
// \ingroup dynamic_matrix
//
// The CUDADynamicMatrix class template is the representation of an arbitrary sized matrix with
// \f$ M \times N \f$ dynamically allocated elements of arbitrary type. The type of the elements
// and the storage order of the matrix can be specified via the two template parameters:

   \code
   template< typename Type, bool SO >
   class CUDADynamicMatrix;
   \endcode

//  - Type: specifies the type of the matrix elements. CUDADynamicMatrix can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - SO  : specifies the storage order (blaze::rowMajor, blaze::columnMajor) of the matrix.
//          The default value is blaze::rowMajor.
//
// Depending on the storage order, the matrix elements are either stored in a row-wise fashion
// or in a column-wise fashion. Given the 2x3 matrix

                          \f[\left(\begin{array}{*{3}{c}}
                          1 & 2 & 3 \\
                          4 & 5 & 6 \\
                          \end{array}\right)\f]\n

// in case of row-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 2 & 3 & 4 & 5 & 6. \\
                          \end{array}\right)\f]

// In case of column-major order the elements are stored in the order

                          \f[\left(\begin{array}{*{6}{c}}
                          1 & 4 & 2 & 5 & 3 & 6. \\
                          \end{array}\right)\f]

// The use of CUDADynamicMatrix is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of row-major and
// column-major dense and sparse matrices with fitting element types. The following example gives
// an impression of the use of CUDADynamicMatrix:

   \code
   using blaze::CUDADynamicMatrix;
   using blaze::CompressedMatrix;
   using blaze::rowMajor;
   using blaze::columnMajor;

   CUDADynamicMatrix<double,rowMajor> A( 2, 3 );  // Default constructed, non-initialized, row-major 2x3 matrix
   A(0,0) = 1.0; A(0,1) = 2.0; A(0,2) = 3.0;  // Initialization of the first row
   A(1,0) = 4.0; A(1,1) = 5.0; A(1,2) = 6.0;  // Initialization of the second row

   CUDADynamicMatrix<float,columnMajor> B( 2, 3 );  // Default constructed column-major single precision 2x3 matrix
   B(0,0) = 1.0; B(0,1) = 3.0; B(0,2) = 5.0;    // Initialization of the first row
   B(1,0) = 2.0; B(1,1) = 4.0; B(1,2) = 6.0;    // Initialization of the second row

   CompressedMatrix<float> C( 2, 3 );        // Empty row-major sparse single precision matrix
   CUDADynamicMatrix<float>    D( 3, 2, 4.0F );  // Directly, homogeneously initialized single precision 3x2 matrix

   CUDADynamicMatrix<double,rowMajor>    E( A );  // Creation of a new row-major matrix as a copy of A
   CUDADynamicMatrix<double,columnMajor> F;       // Creation of a default column-major matrix

   E = A + B;     // Matrix addition and assignment to a row-major matrix
   F = A - C;     // Matrix subtraction and assignment to a column-major matrix
   F = A * D;     // Matrix multiplication between two matrices of different element types

   A *= 2.0;      // In-place scaling of matrix A
   E  = 2.0 * B;  // Scaling of matrix B
   F  = D * 2.0;  // Scaling of matrix D

   E += A - B;    // Addition assignment
   E -= A + C;    // Subtraction assignment
   F *= A * D;    // Multiplication assignment
   \endcode
*/
template< typename Type                    // Data type of the matrix
        , bool SO = defaultStorageOrder >  // Storage order
class CUDADynamicMatrix
   : public DenseMatrix< CUDADynamicMatrix<Type,SO>, SO >
{
 public:
   //**Type definitions****************************************************************************
   using This          = CUDADynamicMatrix<Type,SO>;    //!< Type of this CUDADynamicMatrix instance.
   using BaseType      = DenseMatrix<This,SO>;          //!< Base type of this CUDADynamicMatrix instance.
   using ResultType    = This;                          //!< Result type for expression template evaluations.
   using OppositeType  = CUDADynamicMatrix<Type,!SO>;   //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = CUDADynamicMatrix<Type,!SO>;   //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                          //!< Type of the matrix elements.
   using ReturnType    = const Type&;                   //!< Return type for expression template evaluations.
   using CompositeType = const This&;                   //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant matrix value.
   using ConstReference = const Type&;  //!< Reference to a constant matrix value.
   using Pointer        = Type*;        //!< Pointer to a non-constant matrix value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant matrix value.

   using Iterator      = DenseIterator<Type,usePadding>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,usePadding>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CUDADynamicMatrix with different data/element type.
   */
   template< typename NewType >  // Data type of the other matrix
   struct Rebind {
      using Other = CUDADynamicMatrix<NewType,SO>;  //!< The type of the other CUDADynamicMatrix.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a CUDADynamicMatrix with different fixed dimensions.
   */
   template< size_t NewM    // Number of rows of the other matrix
           , size_t NewN >  // Number of columns of the other matrix
   struct Resize {
      using Other = CUDADynamicMatrix<Type,SO>;  //!< The type of the other CUDADynamicMatrix.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the matrix is involved
       in can be optimized via SIMD operations. In case the element type of the matrix is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   static constexpr bool simdEnabled = false;

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the matrix can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   static constexpr bool smpAssignable = false;

   //! Compilation flag for CUDA assignments.
   /*! The \a cudaAssignable compilation flag indicates whether the matrix can be used in CUDA
       assignments (both on the left-hand and right-hand side of the assignment). */
   static constexpr bool cudaAssignable = !IsCUDAAssignable_v<Type>;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CUDADynamicMatrix() noexcept;
   explicit inline CUDADynamicMatrix( size_t m, size_t n );
   explicit inline CUDADynamicMatrix( size_t m, size_t n, const Type& init );
   explicit inline CUDADynamicMatrix( initializer_list< initializer_list<Type> > list );

   template< typename Other >
   explicit inline CUDADynamicMatrix( size_t m, size_t n, const Other* array );

   template< typename Other, size_t Rows, size_t Cols >
   explicit inline CUDADynamicMatrix( const Other (&array)[Rows][Cols] );

                                     inline CUDADynamicMatrix( const CUDADynamicMatrix& m );
                                     inline CUDADynamicMatrix( CUDADynamicMatrix&& m ) noexcept;
   template< typename MT, bool SO2 > inline CUDADynamicMatrix( const Matrix<MT,SO2>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~CUDADynamicMatrix();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j ) noexcept;
   inline ConstReference operator()( size_t i, size_t j ) const noexcept;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t i ) noexcept;
   inline ConstPointer   data  ( size_t i ) const noexcept;
   inline Iterator       begin ( size_t i ) noexcept;
   inline ConstIterator  begin ( size_t i ) const noexcept;
   inline ConstIterator  cbegin( size_t i ) const noexcept;
   inline Iterator       end   ( size_t i ) noexcept;
   inline ConstIterator  end   ( size_t i ) const noexcept;
   inline ConstIterator  cend  ( size_t i ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CUDADynamicMatrix& operator=( const Type& rhs );
   inline CUDADynamicMatrix& operator=( initializer_list< initializer_list<Type> > list );

   template< typename Other, size_t Rows, size_t Cols >
   inline CUDADynamicMatrix& operator=( const Other (&array)[Rows][Cols] );

   inline CUDADynamicMatrix& operator=( const CUDADynamicMatrix& rhs );
   inline CUDADynamicMatrix& operator=( CUDADynamicMatrix&& rhs ) noexcept;

   template< typename MT, bool SO2 > inline CUDADynamicMatrix& operator= ( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline CUDADynamicMatrix& operator+=( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline CUDADynamicMatrix& operator-=( const Matrix<MT,SO2>& rhs );
   template< typename MT, bool SO2 > inline CUDADynamicMatrix& operator%=( const Matrix<MT,SO2>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t rows() const noexcept;
   inline size_t columns() const noexcept;
   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t i ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t i ) const;
   inline void   reset();
   inline void   reset( size_t i );
   inline void   clear();
          void   resize ( size_t m, size_t n, bool preserve=true );
   inline void   extend ( size_t m, size_t n, bool preserve=true );
   inline void   reserve( size_t elements );
   inline void   shrinkToFit();
   inline void   swap( CUDADynamicMatrix& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline CUDADynamicMatrix& transpose();
   inline CUDADynamicMatrix& ctranspose();

   template< typename Other > inline CUDADynamicMatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedAssign_v = false;
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedAddAssign_v = false;
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedSubAssign_v = false;
   /*! \endcond */
   //**********************************************************************************************

   //**********************************************************************************************
   /*! \cond BLAZE_INTERNAL */
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedSchurAssign_v = false;
   /*! \endcond */
   //**********************************************************************************************

 public:
   //**Debugging functions*************************************************************************
   /*!\name Debugging functions */
   //@{
   inline bool isIntact() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;

   template< typename MT > inline auto assign( const DenseMatrix<MT,SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto assign( const DenseMatrix<MT,!SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto assign( const SparseMatrix<MT,SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto assign( const SparseMatrix<MT,!SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;

   template< typename MT > inline auto addAssign( const DenseMatrix<MT,SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto addAssign( const DenseMatrix<MT,!SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto addAssign( const SparseMatrix<MT,SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto addAssign( const SparseMatrix<MT,!SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;

   template< typename MT > inline auto subAssign( const DenseMatrix<MT,SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto subAssign( const DenseMatrix<MT,!SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto subAssign( const SparseMatrix<MT,SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto subAssign( const SparseMatrix<MT,!SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;

   template< typename MT > inline auto schurAssign( const DenseMatrix<MT,SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto schurAssign( const DenseMatrix<MT,!SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto schurAssign( const SparseMatrix<MT,SO>&  rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto schurAssign( const SparseMatrix<MT,!SO>& rhs )
      -> EnableIf_t< IsCUDAAssignable_v<MT> >;


   template< typename MT > inline auto assign( const DenseMatrix<MT,SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto assign( const DenseMatrix<MT,!SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto assign( const SparseMatrix<MT,SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto assign( const SparseMatrix<MT,!SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;

   template< typename MT > inline auto addAssign( const DenseMatrix<MT,SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto addAssign( const DenseMatrix<MT,!SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto addAssign( const SparseMatrix<MT,SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto addAssign( const SparseMatrix<MT,!SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;

   template< typename MT > inline auto subAssign( const DenseMatrix<MT,SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto subAssign( const DenseMatrix<MT,!SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto subAssign( const SparseMatrix<MT,SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto subAssign( const SparseMatrix<MT,!SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;

   template< typename MT > inline auto schurAssign( const DenseMatrix<MT,SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto schurAssign( const DenseMatrix<MT,!SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto schurAssign( const SparseMatrix<MT,SO>&  rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   template< typename MT > inline auto schurAssign( const SparseMatrix<MT,!SO>& rhs )
      -> DisableIf_t< IsCUDAAssignable_v<MT> >;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t m_;                //!< The current number of rows of the matrix.
   size_t n_;                //!< The current number of columns of the matrix.
   size_t nn_;               //!< The alignment adjusted number of columns.
   size_t capacity_;         //!< The maximum capacity of the matrix.
   Type* BLAZE_RESTRICT v_;  //!< The dynamically allocated matrix elements.
                             /*!< Access to the matrix elements is gained via the function call
                                  operator. In case of row-major order the memory layout of the
                                  elements is
                                  \f[\left(\begin{array}{*{5}{c}}
                                  0            & 1             & 2             & \cdots & N-1         \\
                                  N            & N+1           & N+2           & \cdots & 2 \cdot N-1 \\
                                  \vdots       & \vdots        & \vdots        & \ddots & \vdots      \\
                                  M \cdot N-N  & M \cdot N-N+1 & M \cdot N-N+2 & \cdots & M \cdot N-1 \\
                                  \end{array}\right)\f]. */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The default constructor for CUDADynamicMatrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix() noexcept
   : m_       ( 0UL )      // The current number of rows of the matrix
   , n_       ( 0UL )      // The current number of columns of the matrix
   , nn_      ( 0UL )      // The alignment adjusted number of columns
   , capacity_( 0UL )      // The maximum capacity of the matrix
   , v_       ( nullptr )  // The matrix elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a matrix of size \f$ m \times n \f$. No element initialization is performed!
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
//
// \note This constructor is only responsible to allocate the required dynamic memory. No
// element initialization is performed!
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( size_t m, size_t n )
   : m_       ( m )                             // The current number of rows of the matrix
   , n_       ( n )                             // The current number of columns of the matrix
   , nn_      ( n )                             // The alignment adjusted number of columns
   , capacity_( m_*nn_ )                        // The maximum capacity of the matrix
   , v_       ( cuda_managed_allocate<Type>( capacity_ ) )   // The matrix elements
{
   for( size_t i=0UL; i<m_; ++i ) {
      cuda_transform( begin(i), end(i), begin(i),
         [] BLAZE_DEVICE_CALLABLE ( auto const& ) { return Type(); } );
   }
   cudaDeviceSynchronize();

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogenous initialization of all \f$ m \times n \f$ matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param init The initial value of the matrix elements.
//
// All matrix elements are initialized with the specified value.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( size_t m, size_t n, const Type& init )
   : CUDADynamicMatrix( m, n )
{
   for( size_t i=0UL; i<m_; ++i ) {
      cuda_transform( begin(i), end(i), begin(i),
         [=] BLAZE_DEVICE_CALLABLE ( auto const& ) { return init; } );
   }
   cudaDeviceSynchronize();

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all matrix elements.
//
// \param list The initializer list.
//
// This constructor provides the option to explicitly initialize the elements of the matrix by
// means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::CUDADynamicMatrix<int,rowMajor> A{ { 1, 2, 3 },
                                         { 4, 5 },
                                         { 7, 8, 9 } };
   \endcode

// The matrix is sized according to the size of the initializer list and all its elements are
// (copy) assigned the values of the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example).
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( initializer_list< initializer_list<Type> > list )
   : CUDADynamicMatrix( list.size(), determineColumns( list ) )
{
   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), begin(i) ), end(i), Type() );
      ++i;
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param array Dynamic array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a dynamic array:

   \code
   using blaze::rowMajor;

   int* array = new int[20];
   // ... Initialization of the dynamic array
   blaze::CUDADynamicMatrix<int,rowMajor> v( 4UL, 5UL, array );
   delete[] array;
   \endcode

// The matrix is sized according to the given size of the array and initialized with the values
// from the given array. Note that it is expected that the given \a array has at least \a m by
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type     // Data type of the matrix
        , bool SO >         // Storage order
template< typename Other >  // Data type of the initialization array
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( size_t m, size_t n, const Other* array )
   : CUDADynamicMatrix( m, n )
{
   for( size_t i=0UL; i<m; ++i ) {
      for( size_t j=0UL; j<n; ++j ) {
         v_[i*nn_+j] = array[i*n+j];
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a static array:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CUDADynamicMatrix<int,rowMajor> A( init );
   \endcode

// The matrix is sized according to the size of the array and initialized with the values from
// the given array. Missing values are initialized with default values (as e.g. the value 6 in
// the example).
*/
template< typename Type   // Data type of the matrix
        , bool SO >       // Storage order
template< typename Other  // Data type of the initialization array
        , size_t Rows     // Number of rows of the initialization array
        , size_t Cols >   // Number of columns of the initialization array
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( const Other (&array)[Rows][Cols] )
   : CUDADynamicMatrix( Rows, Cols )
{
   for( size_t i=0UL; i<Rows; ++i ) {
      for( size_t j=0UL; j<Cols; ++j ) {
         v_[i*nn_+j] = array[i][j];
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CUDADynamicMatrix.
//
// \param m Matrix to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( const CUDADynamicMatrix& m )
   : CUDADynamicMatrix( m.m_, m.n_ )
{
   BLAZE_INTERNAL_ASSERT( capacity_ <= m.capacity_, "Invalid capacity estimation" );

   cudaAssign( *this, m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CUDADynamicMatrix.
//
// \param m The matrix to be move into this instance.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( CUDADynamicMatrix&& m ) noexcept
   : m_       ( m.m_        )  // The current number of rows of the matrix
   , n_       ( m.n_        )  // The current number of columns of the matrix
   , nn_      ( m.nn_       )  // The alignment adjusted number of columns
   , capacity_( m.capacity_ )  // The maximum capacity of the matrix
   , v_       ( m.v_        )  // The matrix elements
{
   m.m_        = 0UL;
   m.n_        = 0UL;
   m.nn_       = 0UL;
   m.capacity_ = 0UL;
   m.v_        = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different matrices.
//
// \param m Matrix to be copied.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT    // Type of the foreign matrix
        , bool SO2 >     // Storage order of the foreign matrix
inline CUDADynamicMatrix<Type,SO>::CUDADynamicMatrix( const Matrix<MT,SO2>& m )
   : CUDADynamicMatrix( (~m).rows(), (~m).columns() )
{
   using blaze::assign;

   if( IsSparseMatrix_v<MT> ) {
      reset();
   }

   cudaAssign( *this, ~m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for CUDADynamicMatrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>::~CUDADynamicMatrix()
{
   cuda_managed_deallocate( v_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::Reference
   CUDADynamicMatrix<Type,SO>::operator()( size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstReference
   CUDADynamicMatrix<Type,SO>::operator()( size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i*nn_+j];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::Reference
   CUDADynamicMatrix<Type,SO>::at( size_t i, size_t j )
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstReference
   CUDADynamicMatrix<Type,SO>::at( size_t i, size_t j ) const
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::Pointer
   CUDADynamicMatrix<Type,SO>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a row/column are given by the \c rows() and \c columns() member functions,
// respectively, the total number of elements including padding is given by the \c spacing()
// member function.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstPointer
   CUDADynamicMatrix<Type,SO>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::Pointer
   CUDADynamicMatrix<Type,SO>::data( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return v_ + i*nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the matrix elements of row/column \a i.
//
// \param i The row/column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in row/column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstPointer
   CUDADynamicMatrix<Type,SO>::data( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return v_ + i*nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::Iterator
   CUDADynamicMatrix<Type,SO>::begin( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return Iterator( v_ + i*nn_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstIterator
   CUDADynamicMatrix<Type,SO>::begin( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*nn_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator to the first element of row/column \a i.
//
// This function returns a row/column iterator to the first element of row/column \a i. In case
// the storage order is set to \a rowMajor the function returns an iterator to the first element
// of row \a i, in case the storage flag is set to \a columnMajor the function returns an iterator
// to the first element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstIterator
   CUDADynamicMatrix<Type,SO>::cbegin( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*nn_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::Iterator
   CUDADynamicMatrix<Type,SO>::end( size_t i ) noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return Iterator( v_ + i*nn_ + n_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstIterator
   CUDADynamicMatrix<Type,SO>::end( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*nn_ + n_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of row/column \a i.
//
// \param i The row/column index.
// \return Iterator just past the last element of row/column \a i.
//
// This function returns an row/column iterator just past the last element of row/column \a i.
// In case the storage order is set to \a rowMajor the function returns an iterator just past
// the last element of row \a i, in case the storage flag is set to \a columnMajor the function
// returns an iterator just past the last element of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline typename CUDADynamicMatrix<Type,SO>::ConstIterator
   CUDADynamicMatrix<Type,SO>::cend( size_t i ) const noexcept
{
   BLAZE_USER_ASSERT( i < m_, "Invalid dense matrix row access index" );
   return ConstIterator( v_ + i*nn_ + n_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogenous assignment to all matrix elements.
//
// \param rhs Scalar value to be assigned to all matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator=( const Type& rhs )
{
   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         v_[i*nn_+j] = rhs;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
//
// This assignment operator offers the option to directly assign to all elements of the matrix
// by means of an initializer list:

   \code
   using blaze::rowMajor;

   blaze::CUDADynamicMatrix<int,rowMajor> A;
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The matrix is resized according to the given initializer list and all its elements are
// (copy) assigned the values from the given initializer list. Missing values are initialized
// as default (as e.g. the value 6 in the example).
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>&
   CUDADynamicMatrix<Type,SO>::operator=( initializer_list< initializer_list<Type> > list )
{
   resize( list.size(), determineColumns( list ), false );

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      std::fill( std::copy( rowList.begin(), rowList.end(), v_+i*nn_ ), v_+(i+1UL)*nn_, Type() );
      ++i;
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the assignment.
// \return Reference to the assigned matrix.
//
// This assignment operator offers the option to directly set all elements of the matrix:

   \code
   using blaze::rowMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CUDADynamicMatrix<int,rowMajor> A;
   A = init;
   \endcode

// The matrix is resized according to the size of the array and assigned the values of the given
// array. Missing values are initialized with default values (as e.g. the value 6 in the example).
*/
template< typename Type   // Data type of the matrix
        , bool SO >       // Storage order
template< typename Other  // Data type of the initialization array
        , size_t Rows     // Number of rows of the initialization array
        , size_t Cols >   // Number of columns of the initialization array
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator=( const Other (&array)[Rows][Cols] )
{
   resize( Rows, Cols, false );

   for( size_t i=0UL; i<Rows; ++i )
      for( size_t j=0UL; j<Cols; ++j )
         v_[i*nn_+j] = array[i][j];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for CUDADynamicMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// The matrix is resized according to the given \f$ M \times N \f$ matrix and initialized as a
// copy of this matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator=( const CUDADynamicMatrix& rhs )
{
   if( &rhs == this ) return *this;

   resize( rhs.m_, rhs.n_, false );
   cudaAssign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CUDADynamicMatrix.
//
// \param rhs The matrix to be moved into this instance.
// \return Reference to the assigned matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator=( CUDADynamicMatrix&& rhs ) noexcept
{
   cuda_managed_deallocate( v_ );

   m_        = rhs.m_;
   n_        = rhs.n_;
   nn_       = rhs.nn_;
   capacity_ = rhs.capacity_;
   v_        = rhs.v_;

   rhs.m_        = 0UL;
   rhs.n_        = 0UL;
   rhs.nn_       = 0UL;
   rhs.capacity_ = 0UL;
   rhs.v_        = nullptr;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// The matrix is resized according to the given \f$ M \times N \f$ matrix and initialized as a
// copy of this matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator=( const Matrix<MT,SO2>& rhs )
{
   using blaze::assign;

   using TT = decltype( trans( *this ) );
   using CT = decltype( ctrans( *this ) );
   using IT = decltype( inv( *this ) );

   if( IsSame_v<MT,TT> && (~rhs).isAliased( this ) ) {
      transpose();
   }
   else if( IsSame_v<MT,CT> && (~rhs).isAliased( this ) ) {
      ctranspose();
   }
   else if( !IsSame_v<MT,IT> && (~rhs).canAlias( this ) ) {
      CUDADynamicMatrix tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( (~rhs).rows(), (~rhs).columns(), false );
      if( IsSparseMatrix_v<MT> )
         reset();
      cudaAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator+=( const Matrix<MT,SO2>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      addAssign( *this, tmp );
   }
   else {
      addAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator-=( const Matrix<MT,SO2>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      subAssign( *this, tmp );
   }
   else {
      subAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side matrix for the Schur product.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT    // Type of the right-hand side matrix
        , bool SO2 >     // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::operator%=( const Matrix<MT,SO2>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      schurAssign( *this, tmp );
   }
   else {
      schurAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::rows() const noexcept
{
   return m_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current number of columns of the matrix.
//
// \return The number of columns of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::columns() const noexcept
{
   return n_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the spacing between the beginning of two rows/columns.
//
// \return The spacing between the beginning of two rows/columns.
//
// This function returns the spacing between the beginning of two rows/columns, i.e. the
// total number of elements of a row/column. In case the storage order is set to \a rowMajor
// the function returns the spacing between two rows, in case the storage flag is set to
// \a columnMajor the function returns the spacing between two columns.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::spacing() const noexcept
{
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::capacity() const noexcept
{
   return capacity_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the current capacity of the specified row/column.
//
// \param i The index of the row/column.
// \return The current capacity of row/column \a i.
//
// This function returns the current capacity of the specified row/column. In case the
// storage order is set to \a rowMajor the function returns the capacity of row \a i,
// in case the storage flag is set to \a columnMajor the function returns the capacity
// of column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::capacity( size_t i ) const noexcept
{
   MAYBE_UNUSED( i );
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   return nn_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the dense matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         if( !isDefault( v_[i*nn_+j] ) )
            ++nonzeros;

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the specified row/column.
//
// \param i The index of the row/column.
// \return The number of non-zero elements of row/column \a i.
//
// This function returns the current number of non-zero elements in the specified row/column.
// In case the storage order is set to \a rowMajor the function returns the number of non-zero
// elements in row \a i, in case the storage flag is set to \a columnMajor the function returns
// the number of non-zero elements in column \a i.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline size_t CUDADynamicMatrix<Type,SO>::nonZeros( size_t i ) const
{
   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );

   const size_t jend( i*nn_ + n_ );
   size_t nonzeros( 0UL );

   for( size_t j=i*nn_; j<jend; ++j )
      if( !isDefault( v_[j] ) )
         ++nonzeros;

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::reset()
{
   using blaze::clear;

   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         clear( v_[i*nn_+j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column to the default initial values.
//
// \param i The index of the row/column.
// \return void
//
// This function resets the values in the specified row/column to their default value. In case
// the storage order is set to \a rowMajor the function resets the values in row \a i, in case
// the storage order is set to \a columnMajor the function resets the values in column \a i.
// Note that the capacity of the row/column remains unchanged.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::reset( size_t i )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( i < rows(), "Invalid row access index" );
   for( size_t j=0UL; j<n_; ++j )
      clear( v_[i*nn_+j] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the \f$ M \times N \f$ matrix.
//
// \return void
//
// After the clear() function, the size of the matrix is 0.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::clear()
{
   resize( 0UL, 0UL, false );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the matrix.
//
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function resizes the matrix using the given size to \f$ m \times n \f$. During this
// operation, new dynamic memory may be allocated in case the capacity of the matrix is too
// small. Note that this function may invalidate all existing views (submatrices, rows, columns,
// ...) on the matrix if it is used to shrink the matrix. Additionally, the resize operation
// potentially changes all matrix elements. In order to preserve the old matrix values, the
// \a preserve flag can be set to \a true. However, new matrix elements are not initialized!
//
// The following example illustrates the resize operation of a \f$ 2 \times 4 \f$ matrix to a
// \f$ 4 \times 2 \f$ matrix. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & 3 & 4 \\
                              5 & 6 & 7 & 8 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              5 & 6 \\
                              x & x \\
                              x & x \\
                              \end{array}\right)
                              \f]
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
void CUDADynamicMatrix<Type,SO>::resize( size_t m, size_t n, bool preserve )
{
   using std::swap;
   using blaze::min;

   if( m == m_ && n == n_ ) return;

   const size_t nn( n );

   if( preserve )
   {
      Type* BLAZE_RESTRICT v = cuda_managed_allocate<Type>( m*nn );
      const size_t min_m( min( m, m_ ) );
      const size_t min_n( min( n, n_ ) );

      for( size_t i=0UL; i<min_m; ++i ) {
         transfer( v_+i*nn_, v_+i*nn_+min_n, v+i*nn );
      }

      swap( v_, v );
      cuda_managed_deallocate( v );
      capacity_ = m*nn;
   }
   else if( m*nn > capacity_ ) {
      Type* BLAZE_RESTRICT v = cuda_managed_allocate<Type>( m*nn );
      swap( v_, v );
      cuda_managed_deallocate( v );
      capacity_ = m*nn;
   }

   if( IsVectorizable_v<Type> ) {
      for( size_t i=0UL; i<m; ++i )
         for( size_t j=n; j<nn; ++j )
            v_[i*nn+j] = Type();
   }

   m_  = m;
   n_  = n;
   nn_ = nn;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the matrix.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function increases the matrix size by \a m rows and \a n columns. During this operation,
// new dynamic memory may be allocated in case the capacity of the matrix is too small. Therefore
// this function potentially changes all matrix elements. In order to preserve the old matrix
// values, the \a preserve flag can be set to \a true. However, new matrix elements are not
// initialized!
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::extend( size_t m, size_t n, bool preserve )
{
   resize( m_+m, n_+n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the matrix.
//
// \param elements The new minimum capacity of the dense matrix.
// \return void
//
// This function increases the capacity of the dense matrix to at least \a elements elements.
// The current values of the matrix elements are preserved.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::reserve( size_t elements )
{
   using std::swap;

   if( elements > capacity_ )
   {
      // Allocating a new array
      Type* BLAZE_RESTRICT tmp = cuda_managed_allocate<Type>( elements );

      // Initializing the new array
      transfer( v_, v_+capacity_, tmp );

      if( IsVectorizable_v<Type> ) {
         for( size_t i=capacity_; i<elements; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      swap( tmp, v_ );
      cuda_managed_deallocate( tmp );
      capacity_ = elements;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Requesting the removal of unused capacity.
//
// \return void
//
// This function minimizes the capacity of the matrix by removing unused capacity. Please note
// that due to padding the capacity might not be reduced exactly to rows() times columns().
// Please also note that in case a reallocation occurs, all iterators (including end() iterators),
// all pointers and references to elements of this matrix are invalidated.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::shrinkToFit()
{
   if( ( m_ * nn_ ) < capacity_ ) {
      CUDADynamicMatrix( *this ).swap( *this );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void CUDADynamicMatrix<Type,SO>::swap( CUDADynamicMatrix& m ) noexcept
{
   using std::swap;

   swap( m_ , m.m_  );
   swap( n_ , m.n_  );
   swap( nn_, m.nn_ );
   swap( capacity_, m.capacity_ );
   swap( v_ , m.v_  );
}
//*************************************************************************************************




//=================================================================================================
//
//  NUMERIC FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief In-place transpose of the matrix.
//
// \return Reference to the transposed matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::transpose()
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   if( m_ == n_ )
   {
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( min( ii+block, m_ ) );
         for( size_t jj=0UL; jj<=ii; jj+=block ) {
            for( size_t i=ii; i<iend; ++i ) {
               const size_t jend( min( jj+block, n_, i ) );
               for( size_t j=jj; j<jend; ++j ) {
                  swap( v_[i*nn_+j], v_[j*nn_+i] );
               }
            }
         }
      }
   }
   else
   {
      CUDADynamicMatrix tmp( trans(*this) );
      this->swap( tmp );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief In-place conjugate transpose of the matrix.
//
// \return Reference to the transposed matrix.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::ctranspose()
{
   constexpr size_t block( BLOCK_SIZE );

   if( m_ == n_ )
   {
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( min( ii+block, m_ ) );
         for( size_t jj=0UL; jj<ii; jj+=block ) {
            const size_t jend( min( jj+block, n_ ) );
            for( size_t i=ii; i<iend; ++i ) {
               for( size_t j=jj; j<jend; ++j ) {
                  cswap( v_[i*nn_+j], v_[j*nn_+i] );
               }
            }
         }
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=ii; j<i; ++j ) {
               cswap( v_[i*nn_+j], v_[j*nn_+i] );
            }
            conjugate( v_[i*nn_+i] );
         }
      }
   }
   else
   {
      CUDADynamicMatrix tmp( ctrans(*this) );
      swap( tmp );
   }

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
//
// This function scales the matrix by applying the given scalar value \a scalar to each element
// of the matrix. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   blaze::CUDADynamicMatrix<int> A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the matrix
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type     // Data type of the matrix
        , bool SO >         // Storage order
template< typename Other >  // Data type of the scalar value
inline CUDADynamicMatrix<Type,SO>& CUDADynamicMatrix<Type,SO>::scale( const Other& scalar )
{
   for( size_t i=0UL; i<m_; ++i )
      for( size_t j=0UL; j<n_; ++j )
         v_[i*nn_+j] *= scalar;

   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the dynamic matrix are intact.
//
// \return \a true in case the dynamic matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline bool CUDADynamicMatrix<Type,SO>::isIntact() const noexcept
{
   if( m_ * n_ > capacity_ )
      return false;

   if( IsVectorizable_v<Type> ) {
      for( size_t i=0UL; i<m_; ++i ) {
         for( size_t j=n_; j<nn_; ++j ) {
            if( v_[i*nn_+j] != Type() )
               return false;
         }
      }
   }

   return true;
}
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the matrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address can alias with the matrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the matrix
        , bool SO >         // Storage order
template< typename Other >  // Data type of the foreign expression
inline bool CUDADynamicMatrix<Type,SO>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address is aliased with the matrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the matrix
        , bool SO >         // Storage order
template< typename Other >  // Data type of the foreign expression
inline bool CUDADynamicMatrix<Type,SO>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the matrix can be used in SMP assignments.
//
// \return \a true in case the matrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the matrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline bool CUDADynamicMatrix<Type,SO>::canSMPAssign() const noexcept
{
   return false;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const DenseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         v_[i*nn_+j    ] = (~rhs)(i,j    );
         v_[i*nn_+j+1UL] = (~rhs)(i,j+1UL);
      }
      if( jpos < n_ ) {
         v_[i*nn_+jpos] = (~rhs)(i,jpos);
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const DenseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( min( n_, jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               v_[i*nn_+j] = (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const SparseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*nn_+element->index()] = element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const SparseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*nn_+j] = element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const DenseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal_v<MT> )
      {
         v_[i*nn_+i] += (~rhs)(i,i);
      }
      else
      {
         const size_t jbegin( ( IsUpper_v<MT> )
                              ?( IsStrictlyUpper_v<MT> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend  ( ( IsLower_v<MT> )
                              ?( IsStrictlyLower_v<MT> ? i : i+1UL )
                              :( n_ ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         size_t j( jbegin );

         for( ; (j+2UL) <= jend; j+=2UL ) {
            v_[i*nn_+j    ] += (~rhs)(i,j    );
            v_[i*nn_+j+1UL] += (~rhs)(i,j+1UL);
         }
         if( j < jend ) {
            v_[i*nn_+j] += (~rhs)(i,j);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const DenseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block )
      {
         if( IsLower_v<MT> && ii < jj ) break;
         if( IsUpper_v<MT> && ii > jj ) continue;

         for( size_t i=ii; i<iend; ++i )
         {
            const size_t jbegin( ( IsUpper_v<MT> )
                                 ?( max( ( IsStrictlyUpper_v<MT> ? i+1UL : i ), jj ) )
                                 :( jj ) );
            const size_t jend  ( ( IsLower_v<MT> )
                                 ?( min( ( IsStrictlyLower_v<MT> ? i : i+1UL ), n_, jj+block ) )
                                 :( min( n_, jj+block ) ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               v_[i*nn_+j] += (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const SparseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*nn_+element->index()] += element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const SparseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*nn_+j] += element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const DenseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      if( IsDiagonal_v<MT> )
      {
         v_[i*nn_+i] -= (~rhs)(i,i);
      }
      else
      {
         const size_t jbegin( ( IsUpper_v<MT> )
                              ?( IsStrictlyUpper_v<MT> ? i+1UL : i )
                              :( 0UL ) );
         const size_t jend  ( ( IsLower_v<MT> )
                              ?( IsStrictlyLower_v<MT> ? i : i+1UL )
                              :( n_ ) );
         BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

         size_t j( jbegin );

         for( ; (j+2UL) <= jend; j+=2UL ) {
            v_[i*nn_+j    ] -= (~rhs)(i,j    );
            v_[i*nn_+j+1UL] -= (~rhs)(i,j+1UL);
         }
         if( j < jend ) {
            v_[i*nn_+j] -= (~rhs)(i,j);
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const DenseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block )
      {
         if( IsLower_v<MT> && ii < jj ) break;
         if( IsUpper_v<MT> && ii > jj ) continue;

         for( size_t i=ii; i<iend; ++i )
         {
            const size_t jbegin( ( IsUpper_v<MT> )
                                 ?( max( ( IsStrictlyUpper_v<MT> ? i+1UL : i ), jj ) )
                                 :( jj ) );
            const size_t jend  ( ( IsLower_v<MT> )
                                 ?( min( ( IsStrictlyLower_v<MT> ? i : i+1UL ), n_, jj+block ) )
                                 :( min( n_, jj+block ) ) );
            BLAZE_INTERNAL_ASSERT( jbegin <= jend, "Invalid loop indices detected" );

            for( size_t j=jbegin; j<jend; ++j ) {
               v_[i*nn_+j] -= (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const SparseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i*nn_+element->index()] -= element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const SparseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()*nn_+j] -= element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const DenseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   for( size_t i=0UL; i<m_; ++i ) {
      for( size_t j=0UL; j<jpos; j+=2UL ) {
         v_[i*nn_+j    ] *= (~rhs)(i,j    );
         v_[i*nn_+j+1UL] *= (~rhs)(i,j+1UL);
      }
      if( jpos < n_ ) {
         v_[i*nn_+jpos] *= (~rhs)(i,jpos);
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const DenseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t ii=0UL; ii<m_; ii+=block ) {
      const size_t iend( min( m_, ii+block ) );
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( min( n_, jj+block ) );
         for( size_t i=ii; i<iend; ++i ) {
            for( size_t j=jj; j<jend; ++j ) {
               v_[i*nn_+j] *= (~rhs)(i,j);
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const SparseMatrix<MT,SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::reset;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      size_t j( 0UL );

      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element ) {
         for( ; j<element->index(); ++j )
            reset( v_[i*nn_+j] );
         v_[i*nn_+j] *= element->value();
         ++j;
      }

      for( ; j<n_; ++j ) {
         reset( v_[i*nn_+j] );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const SparseMatrix<MT,!SO>& rhs )
   -> DisableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::reset;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      size_t i( 0UL );

      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element ) {
         for( ; i<element->index(); ++i )
            reset( v_[i*nn_+j] );
         v_[i*nn_+j] *= element->value();
         ++i;
      }

      for( ; i<m_; ++i ) {
         reset( v_[i*nn_+j] );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const DenseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::assign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t jpos( n_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( n_ - ( n_ % 2UL ) ) == jpos, "Invalid end calculation" );

   cudaAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const DenseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const SparseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::assign( const SparseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const DenseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaAddAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaAddAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const DenseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const SparseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaAddAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaAddAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::addAssign( const SparseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const DenseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaSubAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaSubAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const DenseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const SparseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaSubAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaSubAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::subAssign( const SparseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const DenseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaSchurAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaSchurAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const DenseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const SparseMatrix<MT,SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::cudaSchurAssign;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   cudaSchurAssign( *this, ~rhs );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the Schur product assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix for the Schur product.
// \return auto
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
template< typename MT >  // Type of the right-hand side sparse matrix
inline auto CUDADynamicMatrix<Type,SO>::schurAssign( const SparseMatrix<MT,!SO>& rhs )
   -> EnableIf_t< IsCUDAAssignable_v<MT> >
{
   using blaze::reset;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   // TODO
}
//*************************************************************************************************








//=================================================================================================
//
//  CLASS TEMPLATE SPECIALIZATION FOR COLUMN-MAJOR MATRICES
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of CUDADynamicMatrix for column-major matrices.
// \ingroup dynamic_matrix
//
// This specialization of CUDADynamicMatrix adapts the class template to the requirements of
// column-major matrices.
*/
template< typename Type >  // Data type of the matrix
class CUDADynamicMatrix<Type,true>
   : public DenseMatrix< CUDADynamicMatrix<Type,true>, true >
{
 public:
   //**Type definitions****************************************************************************
   using This          = CUDADynamicMatrix<Type,true>;   //!< Type of this CUDADynamicMatrix instance.
   using BaseType      = DenseMatrix<This,true>;     //!< Base type of this CUDADynamicMatrix instance.
   using ResultType    = This;                       //!< Result type for expression template evaluations.
   using OppositeType  = CUDADynamicMatrix<Type,false>;  //!< Result type with opposite storage order for expression template evaluations.
   using TransposeType = CUDADynamicMatrix<Type,false>;  //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                       //!< Type of the matrix elements.
   using ReturnType    = const Type&;                //!< Return type for expression template evaluations.
   using CompositeType = const This&;                //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant matrix value.
   using ConstReference = const Type&;  //!< Reference to a constant matrix value.
   using Pointer        = Type*;        //!< Pointer to a non-constant matrix value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant matrix value.

   using Iterator      = DenseIterator<Type,usePadding>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,usePadding>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CUDADynamicMatrix with different data/element type.
   */
   template< typename NewType >  // Data type of the other matrix
   struct Rebind {
      using Other = CUDADynamicMatrix<NewType,true>;  //!< The type of the other CUDADynamicMatrix.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a CUDADynamicMatrix with different fixed dimensions.
   */
   template< size_t NewM    // Number of rows of the other matrix
           , size_t NewN >  // Number of columns of the other matrix
   struct Resize {
      using Other = CUDADynamicMatrix<Type,true>;  //!< The type of the other CUDADynamicMatrix.
   };
   //**********************************************************************************************

   //**Compilation flags***************************************************************************
   //! Compilation flag for SIMD optimization.
   /*! The \a simdEnabled compilation flag indicates whether expressions the matrix is involved
       in can be optimized via SIMD operations. In case the element type of the matrix is a
       vectorizable data type, the \a simdEnabled compilation flag is set to \a true, otherwise
       it is set to \a false. */
   static constexpr bool simdEnabled = false;

   //! Compilation flag for SMP assignments.
   /*! The \a smpAssignable compilation flag indicates whether the matrix can be used in SMP
       (shared memory parallel) assignments (both on the left-hand and right-hand side of the
       assignment). */
   static constexpr bool smpAssignable = false;
   //**********************************************************************************************

   //! Compilation flag for CUDA assignments.
   /*! The \a cudaAssignable compilation flag indicates whether the matrix can be used in CUDA
       assignments (both on the left-hand and right-hand side of the assignment). */
   static constexpr bool cudaAssignable = !IsCUDAAssignable_v<Type>;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CUDADynamicMatrix() noexcept;
   explicit inline CUDADynamicMatrix( size_t m, size_t n );
   explicit inline CUDADynamicMatrix( size_t m, size_t n, const Type& init );
   explicit inline CUDADynamicMatrix( initializer_list< initializer_list<Type> > list );

   template< typename Other > explicit inline CUDADynamicMatrix( size_t m, size_t n, const Other* array );

   template< typename Other, size_t Rows, size_t Cols >
   explicit inline CUDADynamicMatrix( const Other (&array)[Rows][Cols] );

                                    inline CUDADynamicMatrix( const CUDADynamicMatrix& m );
                                    inline CUDADynamicMatrix( CUDADynamicMatrix&& m );
   template< typename MT, bool SO > inline CUDADynamicMatrix( const Matrix<MT,SO>& m );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~CUDADynamicMatrix();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline Reference      operator()( size_t i, size_t j ) noexcept;
   inline ConstReference operator()( size_t i, size_t j ) const noexcept;
   inline Reference      at( size_t i, size_t j );
   inline ConstReference at( size_t i, size_t j ) const;
   inline Pointer        data  () noexcept;
   inline ConstPointer   data  () const noexcept;
   inline Pointer        data  ( size_t j ) noexcept;
   inline ConstPointer   data  ( size_t j ) const noexcept;
   inline Iterator       begin ( size_t j ) noexcept;
   inline ConstIterator  begin ( size_t j ) const noexcept;
   inline ConstIterator  cbegin( size_t j ) const noexcept;
   inline Iterator       end   ( size_t j ) noexcept;
   inline ConstIterator  end   ( size_t j ) const noexcept;
   inline ConstIterator  cend  ( size_t j ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CUDADynamicMatrix& operator=( const Type& rhs );
   inline CUDADynamicMatrix& operator=( initializer_list< initializer_list<Type> > list );

   template< typename Other, size_t Rows, size_t Cols >
   inline CUDADynamicMatrix& operator=( const Other (&array)[Rows][Cols] );

   inline CUDADynamicMatrix& operator=( const CUDADynamicMatrix& rhs );
   inline CUDADynamicMatrix& operator=( CUDADynamicMatrix&& rhs );

   template< typename MT, bool SO > inline CUDADynamicMatrix& operator= ( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline CUDADynamicMatrix& operator+=( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline CUDADynamicMatrix& operator-=( const Matrix<MT,SO>& rhs );
   template< typename MT, bool SO > inline CUDADynamicMatrix& operator%=( const Matrix<MT,SO>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t rows() const noexcept;
   inline size_t columns() const noexcept;
   inline size_t spacing() const noexcept;
   inline size_t capacity() const noexcept;
   inline size_t capacity( size_t j ) const noexcept;
   inline size_t nonZeros() const;
   inline size_t nonZeros( size_t j ) const;
   inline void   reset();
   inline void   reset( size_t j );
   inline void   clear();
          void   resize ( size_t m, size_t n, bool preserve=true );
   inline void   extend ( size_t m, size_t n, bool preserve=true );
   inline void   reserve( size_t elements );
   inline void   shrinkToFit();
   inline void   swap( CUDADynamicMatrix& m ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   inline CUDADynamicMatrix& transpose();
   inline CUDADynamicMatrix& ctranspose();

   template< typename Other > inline CUDADynamicMatrix& scale( const Other& scalar );
   //@}
   //**********************************************************************************************

 private:
   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedAssign_v = false;
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedAddAssign_v = false;
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedSubAssign_v = false;
   //**********************************************************************************************

   //**********************************************************************************************
   //! Helper variable template for the explicit application of the SFINAE principle.
   template< typename MT >
   static constexpr bool VectorizedSchurAssign_v = false;
   //**********************************************************************************************

 public:
   //**Debugging functions*************************************************************************
   /*!\name Debugging functions */
   //@{
   inline bool isIntact() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   template< typename Other > inline bool canAlias ( const Other* alias ) const noexcept;
   template< typename Other > inline bool isAliased( const Other* alias ) const noexcept;

   inline bool canSMPAssign() const noexcept;

   template< typename MT > inline auto assign( const DenseMatrix<MT,true>& rhs );
   template< typename MT > inline void assign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void assign( const SparseMatrix<MT,false>& rhs );

   template< typename MT > inline auto addAssign( const DenseMatrix<MT,true>& rhs );
   template< typename MT > inline void addAssign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void addAssign( const SparseMatrix<MT,false>& rhs );

   template< typename MT > inline auto subAssign ( const DenseMatrix<MT,true>& rhs );
   template< typename MT > inline void subAssign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void subAssign( const SparseMatrix<MT,false>& rhs );

   template< typename MT > inline auto schurAssign ( const DenseMatrix<MT,true>& rhs );
   template< typename MT > inline void schurAssign( const DenseMatrix<MT,false>&  rhs );
   template< typename MT > inline void schurAssign( const SparseMatrix<MT,true>&  rhs );
   template< typename MT > inline void schurAssign( const SparseMatrix<MT,false>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t m_;                //!< The current number of rows of the matrix.
   size_t mm_;               //!< The alignment adjusted number of rows.
   size_t n_;                //!< The current number of columns of the matrix.
   size_t capacity_;         //!< The maximum capacity of the matrix.
   Type* BLAZE_RESTRICT v_;  //!< The dynamically allocated matrix elements.
                             /*!< Access to the matrix elements is gained via the function call
                                  operator. */
   //@}
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   BLAZE_CONSTRAINT_MUST_NOT_BE_POINTER_TYPE  ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_REFERENCE_TYPE( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_CONST         ( Type );
   BLAZE_CONSTRAINT_MUST_NOT_BE_VOLATILE      ( Type );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The default constructor for CUDADynamicMatrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix() noexcept
   : m_       ( 0UL )      // The current number of rows of the matrix
   , mm_      ( 0UL )      // The alignment adjusted number of rows
   , n_       ( 0UL )      // The current number of columns of the matrix
   , capacity_( 0UL )      // The maximum capacity of the matrix
   , v_       ( nullptr )  // The matrix elements
{}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a matrix of size \f$ m \times n \f$. No element initialization is performed!
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
//
// \note This constructor is only responsible to allocate the required dynamic memory. No
// element initialization is performed!
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( size_t m, size_t n )
   : m_       ( m )                            // The current number of rows of the matrix
   , mm_      ( m )              // The alignment adjusted number of rows
   , n_       ( n )                            // The current number of columns of the matrix
   , capacity_( mm_*n_ )                       // The maximum capacity of the matrix
   , v_       ( cuda_managed_allocate<Type>( capacity_ ) )  // The matrix elements
{
   if( IsVectorizable_v<Type> ) {
      for( size_t j=0UL; j<n_; ++j ) {
         for( size_t i=m_; i<mm_; ++i ) {
            v_[i+j*mm_] = Type();
         }
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Constructor for a homogenous initialization of all \f$ m \times n \f$ matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param init The initial value of the matrix elements.
//
// All matrix elements are initialized with the specified value.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( size_t m, size_t n, const Type& init )
   : CUDADynamicMatrix( m, n )
{
   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<m_; ++i ) {
         v_[i+j*mm_] = init;
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List initialization of all matrix elements.
//
// \param list The initializer list.
//
// This constructor provides the option to explicitly initialize the elements of the matrix by
// means of an initializer list:

   \code
   using blaze::columnMajor;

   blaze::CUDADynamicMatrix<int,columnMajor> A{ { 1, 2, 3 },
                                            { 4, 5 },
                                            { 7, 8, 9 } };
   \endcode

// The matrix is sized according to the size of the initializer list and all its elements are
// (copy) assigned the values of the given initializer list. Missing values are initialized as
// default (as e.g. the value 6 in the example).
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( initializer_list< initializer_list<Type> > list )
   : CUDADynamicMatrix( list.size(), determineColumns( list ) )
{
   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         v_[i+j*mm_] = element;
         ++j;
      }
      for( ; j<n_; ++j ) {
         v_[i+j*mm_] = Type();
      }
      ++i;
   }

   BLAZE_INTERNAL_ASSERT( i == m_, "Invalid number of elements detected" );
   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array initialization of all matrix elements.
//
// \param m The number of rows of the matrix.
// \param n The number of columns of the matrix.
// \param array Dynamic array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a dynamic array:

   \code
   using blaze::columnMajor;

   int* array = new int[20];
   // ... Initialization of the dynamic array
   blaze::CUDADynamicMatrix<int,columnMajor> v( array, 5UL, 4UL );
   delete[] array;
   \endcode

// The matrix is sized according to the given size of the array and initialized with the values
// from the given array. Note that it is expected that the given \a array has at least \a m by
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type >   // Data type of the matrix
template< typename Other >  // Data type of the initialization array
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( size_t m, size_t n, const Other* array )
   : CUDADynamicMatrix( m, n )
{
   for( size_t j=0UL; j<n; ++j ) {
      for( size_t i=0UL; i<m; ++i ) {
         v_[i+j*mm_] = array[i+j*m];
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array initialization of all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the matrix with
// a static array:

   \code
   using blaze::columnMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CUDADynamicMatrix<int,columnMajor> A( init );
   \endcode

// The matrix is sized according to the size of the array and initialized with the values from
// the given array. Missing values are initialized with default values (as e.g. the value 6 in
// the example).
*/
template< typename Type >  // Data type of the matrix
template< typename Other   // Data type of the initialization array
        , size_t Rows      // Number of rows of the initialization array
        , size_t Cols >    // Number of columns of the initialization array
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( const Other (&array)[Rows][Cols] )
   : CUDADynamicMatrix( Rows, Cols )
{
   for( size_t j=0UL; j<Cols; ++j ) {
      for( size_t i=0UL; i<Rows; ++i ) {
         v_[i+j*mm_] = array[i][j];
      }
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The copy constructor for CUDADynamicMatrix.
//
// \param m Matrix to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( const CUDADynamicMatrix& m )
   : CUDADynamicMatrix( m.m_, m.n_ )
{
   BLAZE_INTERNAL_ASSERT( capacity_ <= m.capacity_, "Invalid capacity estimation" );

  cudaAassign( *this, m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The move constructor for CUDADynamicMatrix.
//
// \param m The matrix to be moved into this instance.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( CUDADynamicMatrix&& m )
   : m_       ( m.m_        )  // The current number of rows of the matrix
   , mm_      ( m.mm_       )  // The alignment adjusted number of rows
   , n_       ( m.n_        )  // The current number of columns of the matrix
   , capacity_( m.capacity_ )  // The maximum capacity of the matrix
   , v_       ( m.v_        )  // The matrix elements
{
   m.m_        = 0UL;
   m.mm_       = 0UL;
   m.n_        = 0UL;
   m.capacity_ = 0UL;
   m.v_        = nullptr;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Conversion constructor from different matrices.
//
// \param m Matrix to be copied.
*/
template< typename Type >  // Data type of the matrix
template< typename MT      // Type of the foreign matrix
        , bool SO >        // Storage order of the foreign matrix
inline CUDADynamicMatrix<Type,true>::CUDADynamicMatrix( const Matrix<MT,SO>& m )
   : CUDADynamicMatrix( (~m).rows(), (~m).columns() )
{
   if( IsSparseMatrix_v<MT> ) {
      reset();
   }

   cudaAssign( *this, ~m );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief The destructor for CUDADynamicMatrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>::~CUDADynamicMatrix()
{
   cuda_managed_deallocate( v_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::Reference
   CUDADynamicMatrix<Type,true>::operator()( size_t i, size_t j ) noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i+j*mm_];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief 2D-access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access indices.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstReference
   CUDADynamicMatrix<Type,true>::operator()( size_t i, size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( i<m_, "Invalid row access index"    );
   BLAZE_USER_ASSERT( j<n_, "Invalid column access index" );
   return v_[i+j*mm_];
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::Reference
   CUDADynamicMatrix<Type,true>::at( size_t i, size_t j )
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Checked access to the matrix elements.
//
// \param i Access index for the row. The index has to be in the range \f$[0..M-1]\f$.
// \param j Access index for the column. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid matrix access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access indices.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstReference
   CUDADynamicMatrix<Type,true>::at( size_t i, size_t j ) const
{
   if( i >= m_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid row access index" );
   }
   if( j >= n_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid column access index" );
   }
   return (*this)(i,j);
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a column are given by the \c columns() member functions, the total number
// of elements including padding is given by the \c spacing() member function.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::Pointer
   CUDADynamicMatrix<Type,true>::data() noexcept
{
   return v_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic matrix. Note that you
// can NOT assume that all matrix elements lie adjacent to each other! The dynamic matrix may
// use techniques such as padding to improve the alignment of the data. Whereas the number of
// elements within a column are given by the \c columns() member functions, the total number
// of elements including padding is given by the \c spacing() member function.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstPointer
   CUDADynamicMatrix<Type,true>::data() const noexcept
{
   return v_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::Pointer
   CUDADynamicMatrix<Type,true>::data( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return v_ + j*mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Low-level data access to the matrix elements of column \a j.
//
// \param j The column index.
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage for the elements in column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstPointer
   CUDADynamicMatrix<Type,true>::data( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return v_ + j*mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of column \a j.
//
// \param j The column index.
// \return Iterator to the first element of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::Iterator
   CUDADynamicMatrix<Type,true>::begin( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return Iterator( v_ + j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of column \a j.
//
// \param j The column index.
// \return Iterator to the first element of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstIterator
   CUDADynamicMatrix<Type,true>::begin( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator to the first element of column \a j.
//
// \param j The column index.
// \return Iterator to the first element of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstIterator
   CUDADynamicMatrix<Type,true>::cbegin( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*mm_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last element of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::Iterator
   CUDADynamicMatrix<Type,true>::end( size_t j ) noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return Iterator( v_ + j*mm_ + m_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last element of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstIterator
   CUDADynamicMatrix<Type,true>::end( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*mm_ + m_ );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns an iterator just past the last element of column \a j.
//
// \param j The column index.
// \return Iterator just past the last element of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline typename CUDADynamicMatrix<Type,true>::ConstIterator
   CUDADynamicMatrix<Type,true>::cend( size_t j ) const noexcept
{
   BLAZE_USER_ASSERT( j < n_, "Invalid dense matrix column access index" );
   return ConstIterator( v_ + j*mm_ + m_ );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Homogenous assignment to all matrix elements.
//
// \param rhs Scalar value to be assigned to all matrix elements.
// \return Reference to the assigned matrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator=( const Type& rhs )
{
   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*mm_] = rhs;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief List assignment to all matrix elements.
//
// \param list The initializer list.
//
// This assignment operator offers the option to directly assign to all elements of the matrix
// by means of an initializer list:

   \code
   using blaze::columnMajor;

   blaze::CUDADynamicMatrix<int,columnMajor> A;
   A = { { 1, 2, 3 },
         { 4, 5 },
         { 7, 8, 9 } };
   \endcode

// The matrix is resized according to the given initializer list and all its elements are
// (copy) assigned the values from the given initializer list. Missing values are initialized
// as default (as e.g. the value 6 in the example).
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>&
   CUDADynamicMatrix<Type,true>::operator=( initializer_list< initializer_list<Type> > list )
{
   resize( list.size(), determineColumns( list ), false );

   size_t i( 0UL );

   for( const auto& rowList : list ) {
      size_t j( 0UL );
      for( const auto& element : rowList ) {
         v_[i+j*mm_] = element;
         ++j;
      }
      for( ; j<n_; ++j ) {
         v_[i+j*mm_] = Type();
      }
      ++i;
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Array assignment to all matrix elements.
//
// \param array \f$ M \times N \f$ dimensional array for the assignment.
// \return Reference to the assigned matrix.
//
// This assignment operator offers the option to directly set all elements of the matrix:

   \code
   using blaze::columnMajor;

   const int init[3][3] = { { 1, 2, 3 },
                            { 4, 5 },
                            { 7, 8, 9 } };
   blaze::CUDADynamicMatrix<int,columnMajor> A;
   A = init;
   \endcode

// The matrix is resized according to the size of the array and assigned the values of the given
// array. Missing values are initialized with default values (as e.g. the value 6 in the example).
*/
template< typename Type >  // Data type of the matrix
template< typename Other   // Data type of the initialization array
        , size_t Rows      // Number of rows of the initialization array
        , size_t Cols >    // Number of columns of the initialization array
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator=( const Other (&array)[Rows][Cols] )
{
   resize( Rows, Cols, false );

   for( size_t j=0UL; j<Cols; ++j )
      for( size_t i=0UL; i<Rows; ++i )
         v_[i+j*mm_] = array[i][j];

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Copy assignment operator for CUDADynamicMatrix.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// The matrix is resized according to the given \f$ M \times N \f$ matrix and initialized as a
// copy of this matrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator=( const CUDADynamicMatrix& rhs )
{
   using blaze::assign;

   if( &rhs == this ) return *this;

   resize( rhs.m_, rhs.n_, false );
   cudaAssign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Move assignment operator for CUDADynamicMatrix.
//
// \param rhs The matrix to be moved into this instance.
// \return Reference to the assigned matrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator=( CUDADynamicMatrix&& rhs )
{
   cuda_managed_deallocate( v_ );

   m_        = rhs.m_;
   mm_       = rhs.mm_;
   n_        = rhs.n_;
   capacity_ = rhs.capacity_;
   v_        = rhs.v_;

   rhs.m_        = 0UL;
   rhs.mm_       = 0UL;
   rhs.n_        = 0UL;
   rhs.capacity_ = 0UL;
   rhs.v_        = nullptr;

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Assignment operator for different matrices.
//
// \param rhs Matrix to be copied.
// \return Reference to the assigned matrix.
//
// The matrix is resized according to the given \f$ M \times N \f$ matrix and initialized as a
// copy of this matrix.
*/
template< typename Type >  // Data type of the matrix
template< typename MT      // Type of the right-hand side matrix
        , bool SO >        // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator=( const Matrix<MT,SO>& rhs )
{
   using blaze::assign;

   using TT = decltype( trans( *this ) );
   using CT = decltype( ctrans( *this ) );
   using IT = decltype( inv( *this ) );

   if( IsSame_v<MT,TT> && (~rhs).isAliased( this ) ) {
      transpose();
   }
   else if( IsSame_v<MT,CT> && (~rhs).isAliased( this ) ) {
      ctranspose();
   }
   else if( !IsSame_v<MT,IT> && (~rhs).canAlias( this ) ) {
      CUDADynamicMatrix tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( (~rhs).rows(), (~rhs).columns(), false );
      if( IsSparseMatrix_v<MT> )
         reset();
      cudaAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Addition assignment operator for the addition of a matrix (\f$ A+=B \f$).
//
// \param rhs The right-hand side matrix to be added to the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type >  // Data type of the matrix
template< typename MT      // Type of the right-hand side matrix
        , bool SO >        // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator+=( const Matrix<MT,SO>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      addAssign( *this, tmp );
   }
   else {
      addAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Subtraction assignment operator for the subtraction of a matrix (\f$ A-=B \f$).
//
// \param rhs The right-hand side matrix to be subtracted from the matrix.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type >  // Data type of the matrix
template< typename MT      // Type of the right-hand side matrix
        , bool SO >        // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator-=( const Matrix<MT,SO>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      subAssign( *this, tmp );
   }
   else {
      subAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Schur product assignment operator for the multiplication of a matrix (\f$ A\circ=B \f$).
//
// \param rhs The right-hand side matrix for the Schur product.
// \return Reference to the matrix.
// \exception std::invalid_argument Matrix sizes do not match.
//
// In case the current sizes of the two matrices don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type >  // Data type of the matrix
template< typename MT      // Type of the right-hand side matrix
        , bool SO >        // Storage order of the right-hand side matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::operator%=( const Matrix<MT,SO>& rhs )
{
   if( (~rhs).rows() != m_ || (~rhs).columns() != n_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Matrix sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<MT> tmp( ~rhs );
      schurAssign( *this, tmp );
   }
   else {
      schurAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of rows of the matrix.
//
// \return The number of rows of the matrix.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::rows() const noexcept
{
   return m_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current number of columns of the matrix.
//
// \return The number of columns of the matrix.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::columns() const noexcept
{
   return n_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the spacing between the beginning of two columns.
//
// \return The spacing between the beginning of two columns.
//
// This function returns the spacing between the beginning of two column, i.e. the total number
// of elements of a column.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::spacing() const noexcept
{
   return mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the maximum capacity of the matrix.
//
// \return The capacity of the matrix.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::capacity() const noexcept
{
   return capacity_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the current capacity of the specified column.
//
// \param j The index of the column.
// \return The current capacity of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::capacity( size_t j ) const noexcept
{
   MAYBE_UNUSED( j );
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   return mm_;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the total number of non-zero elements in the matrix
//
// \return The number of non-zero elements in the dense matrix.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::nonZeros() const
{
   size_t nonzeros( 0UL );

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         if( !isDefault( v_[i+j*mm_] ) )
            ++nonzeros;

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns the number of non-zero elements in the specified column.
//
// \param j The index of the column.
// \return The number of non-zero elements of column \a j.
*/
template< typename Type >  // Data type of the matrix
inline size_t CUDADynamicMatrix<Type,true>::nonZeros( size_t j ) const
{
   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );

   const size_t iend( j*mm_ + m_ );
   size_t nonzeros( 0UL );

   for( size_t i=j*mm_; i<iend; ++i )
      if( !isDefault( v_[i] ) )
         ++nonzeros;

   return nonzeros;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::reset()
{
   using blaze::clear;

   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         clear( v_[i+j*mm_] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Reset the specified column to the default initial values.
//
// \param j The index of the column.
// \return void
//
// This function reset the values in the specified column to their default value. Note that
// the capacity of the column remains unchanged.
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::reset( size_t j )
{
   using blaze::clear;

   BLAZE_USER_ASSERT( j < columns(), "Invalid column access index" );
   for( size_t i=0UL; i<m_; ++i )
      clear( v_[i+j*mm_] );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Clearing the \f$ M \times N \f$ matrix.
//
// \return void
//
// After the clear() function, the size of the matrix is 0.
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::clear()
{
   resize( 0UL, 0UL, false );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Changing the size of the matrix.
//
// \param m The new number of rows of the matrix.
// \param n The new number of columns of the matrix.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function resizes the matrix using the given size to \f$ m \times n \f$. During this
// operation, new dynamic memory may be allocated in case the capacity of the matrix is too
// small. Note that this function may invalidate all existing views (submatrices, rows, columns,
// ...) on the matrix if it is used to shrink the matrix. Additionally, the resize operation
// potentially changes all matrix elements. In order to preserve the old matrix values, the
// \a preserve flag can be set to \a true. However, new matrix elements are not initialized!
//
// The following example illustrates the resize operation of a \f$ 2 \times 4 \f$ matrix to a
// \f$ 4 \times 2 \f$ matrix. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & 3 & 4 \\
                              5 & 6 & 7 & 8 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              5 & 6 \\
                              x & x \\
                              x & x \\
                              \end{array}\right)
                              \f]
*/
template< typename Type >  // Data type of the matrix
void CUDADynamicMatrix<Type,true>::resize( size_t m, size_t n, bool preserve )
{
   using std::swap;
   using blaze::min;

   if( m == m_ && n == n_ ) return;

   const size_t mm( m );

   if( preserve )
   {
      Type* BLAZE_RESTRICT v = cuda_managed_allocate<Type>( mm*n );
      const size_t min_m( min( m, m_ ) );
      const size_t min_n( min( n, n_ ) );

      for( size_t j=0UL; j<min_n; ++j ) {
         transfer( v_+j*mm_, v_+min_m+j*mm_, v+j*mm );
      }

      swap( v_, v );
      cuda_managed_deallocate( v );
      capacity_ = mm*n;
   }
   else if( mm*n > capacity_ ) {
      Type* BLAZE_RESTRICT v = cuda_managed_allocate<Type>( mm*n );
      swap( v_, v );
      cuda_managed_deallocate( v );
      capacity_ = mm*n;
   }

   if( IsVectorizable_v<Type> ) {
      for( size_t j=0UL; j<n; ++j )
         for( size_t i=m; i<mm; ++i )
            v_[i+j*mm] = Type();
   }

   m_  = m;
   mm_ = mm;
   n_  = n;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Extending the size of the matrix.
//
// \param m Number of additional rows.
// \param n Number of additional columns.
// \param preserve \a true if the old values of the matrix should be preserved, \a false if not.
// \return void
//
// This function increases the matrix size by \a m rows and \a n columns. During this operation,
// new dynamic memory may be allocated in case the capacity of the matrix is too small. Therefore
// this function potentially changes all matrix elements. In order to preserve the old matrix
// values, the \a preserve flag can be set to \a true. However, new matrix elements are not
// initialized!
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::extend( size_t m, size_t n, bool preserve )
{
   resize( m_+m, n_+n, preserve );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Setting the minimum capacity of the matrix.
//
// \param elements The new minimum capacity of the dense matrix.
// \return void
//
// This function increases the capacity of the dense matrix to at least \a elements elements.
// The current values of the matrix elements are preserved.
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::reserve( size_t elements )
{
   using std::swap;

   if( elements > capacity_ )
   {
      // Allocating a new array
      Type* BLAZE_RESTRICT tmp = cuda_managed_allocate<Type>( elements );

      // Initializing the new array
      transfer( v_, v_+capacity_, tmp );

      if( IsVectorizable_v<Type> ) {
         for( size_t i=capacity_; i<elements; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      swap( tmp, v_ );
      cuda_managed_deallocate( tmp );
      capacity_ = elements;
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Requesting the removal of unused capacity.
//
// \return void
//
// This function minimizes the capacity of the matrix by removing unused capacity. Please note
// that due to padding the capacity might not be reduced exactly to rows() times columns().
// Please also note that in case a reallocation occurs, all iterators (including end() iterators),
// all pointers and references to elements of this matrix are invalidated.
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::shrinkToFit()
{
   if( ( mm_ * n_ ) < capacity_ ) {
      CUDADynamicMatrix( *this ).swap( *this );
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Swapping the contents of two matrices.
//
// \param m The matrix to be swapped.
// \return void
*/
template< typename Type >  // Data type of the matrix
inline void CUDADynamicMatrix<Type,true>::swap( CUDADynamicMatrix& m ) noexcept
{
   using std::swap;

   swap( m_ , m.m_  );
   swap( mm_, m.mm_ );
   swap( n_ , m.n_  );
   swap( capacity_, m.capacity_ );
   swap( v_ , m.v_  );
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  NUMERIC FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place transpose of the matrix.
//
// \return Reference to the transposed matrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::transpose()
{
   using std::swap;

   constexpr size_t block( BLOCK_SIZE );

   if( m_ == n_ )
   {
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( min( jj+block, n_ ) );
         for( size_t ii=0UL; ii<=jj; ii+=block ) {
            for( size_t j=jj; j<jend; ++j ) {
               const size_t iend( min( ii+block, m_, j ) );
               for( size_t i=ii; i<iend; ++i ) {
                  swap( v_[i+j*mm_], v_[j+i*mm_] );
               }
            }
         }
      }
   }
   else
   {
      CUDADynamicMatrix tmp( trans(*this) );
      this->swap( tmp );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief In-place conjugate transpose of the matrix.
//
// \return Reference to the transposed matrix.
*/
template< typename Type >  // Data type of the matrix
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::ctranspose()
{
   constexpr size_t block( BLOCK_SIZE );

   if( m_ == n_ )
   {
      for( size_t jj=0UL; jj<n_; jj+=block ) {
         const size_t jend( min( jj+block, n_ ) );
         for( size_t ii=0UL; ii<jj; ii+=block ) {
            const size_t iend( min( ii+block, m_ ) );
            for( size_t j=jj; j<jend; ++j ) {
               for( size_t i=ii; i<iend; ++i ) {
                  cswap( v_[i+j*mm_], v_[j+i*mm_] );
               }
            }
         }
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=jj; i<j; ++i ) {
               cswap( v_[i+j*mm_], v_[j+i*mm_] );
            }
            conjugate( v_[j+j*mm_] );
         }
      }
   }
   else
   {
      CUDADynamicMatrix tmp( ctrans(*this) );
      this->swap( tmp );
   }

   return *this;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Scaling of the matrix by the scalar value \a scalar (\f$ A=B*s \f$).
//
// \param scalar The scalar value for the matrix scaling.
// \return Reference to the matrix.
//
// This function scales the matrix by applying the given scalar value \a scalar to each element
// of the matrix. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   blaze::CUDADynamicMatrix<int> A;
   // ... Resizing and initialization
   A *= 4;        // Scaling of the matrix
   A.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type >   // Data type of the matrix
template< typename Other >  // Data type of the scalar value
inline CUDADynamicMatrix<Type,true>& CUDADynamicMatrix<Type,true>::scale( const Other& scalar )
{
   for( size_t j=0UL; j<n_; ++j )
      for( size_t i=0UL; i<m_; ++i )
         v_[i+j*mm_] *= scalar;

   return *this;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the invariants of the dynamic matrix are intact.
//
// \return \a true in case the dynamic matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type >  // Data type of the matrix
inline bool CUDADynamicMatrix<Type,true>::isIntact() const noexcept
{
   if( m_ * n_ > capacity_ )
      return false;

   if( IsVectorizable_v<Type> ) {
      for( size_t j=0UL; j<n_; ++j ) {
         for( size_t i=m_; i<mm_; ++i ) {
            if( v_[i+j*mm_] != Type() )
               return false;
         }
      }
   }

   return true;
}
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address can alias with the matrix. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type >   // Data type of the matrix
template< typename Other >  // Data type of the foreign expression
inline bool CUDADynamicMatrix<Type,true>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this matrix, \a false if not.
//
// This function returns whether the given address is aliased with the matrix. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type >   // Data type of the matrix
template< typename Other >  // Data type of the foreign expression
inline bool CUDADynamicMatrix<Type,true>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Returns whether the matrix can be used in SMP assignments.
//
// \return \a true in case the matrix can be used in SMP assignments, \a false if not.
//
// This function returns whether the matrix can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current number of
// rows and/or columns of the matrix).
*/
template< typename Type >  // Data type of the matrix
inline bool CUDADynamicMatrix<Type,true>::canSMPAssign() const noexcept
{
   return false;
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,true>::assign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; i<ipos; i+=2UL ) {
         v_[i    +j*mm_] = (~rhs)(i    ,j);
         v_[i+1UL+j*mm_] = (~rhs)(i+1UL,j);
      }
      if( ipos < m_ ) {
         v_[ipos+j*mm_] = (~rhs)(ipos,j);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline void CUDADynamicMatrix<Type,true>::assign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( min( m_, ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               v_[i+j*mm_] = (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::assign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<(~rhs).columns(); ++j )
      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*mm_] = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::assign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<(~rhs).rows(); ++i )
      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*mm_] = element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,true>::addAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal_v<MT> )
      {
         v_[j+j*mm_] += (~rhs)(j,j);
      }
      else
      {
         const size_t ibegin( ( IsLower_v<MT> )
                              ?( IsStrictlyLower_v<MT> ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend  ( ( IsUpper_v<MT> )
                              ?( IsStrictlyUpper_v<MT> ? j : j+1UL )
                              :( m_ ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         size_t i( ibegin );

         for( ; (i+2UL) <= iend; i+=2UL ) {
            v_[i    +j*mm_] += (~rhs)(i    ,j);
            v_[i+1UL+j*mm_] += (~rhs)(i+1UL,j);
         }
         if( i < iend ) {
            v_[i+j*mm_] += (~rhs)(i,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline void CUDADynamicMatrix<Type,true>::addAssign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block )
      {
         if( IsLower_v<MT> && ii < jj ) continue;
         if( IsUpper_v<MT> && ii > jj ) break;

         for( size_t j=jj; j<jend; ++j )
         {
            const size_t ibegin( ( IsLower_v<MT> )
                                 ?( max( ( IsStrictlyLower_v<MT> ? j+1UL : j ), ii ) )
                                 :( ii ) );
            const size_t iend  ( ( IsUpper_v<MT> )
                                 ?( min( ( IsStrictlyUpper_v<MT> ? j : j+1UL ), m_, ii+block ) )
                                 :( min( m_, ii+block ) ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               v_[i+j*mm_] += (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::addAssign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<(~rhs).columns(); ++j )
      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*mm_] += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the addition assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::addAssign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<(~rhs).rows(); ++i )
      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*mm_] += element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,true>::subAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      if( IsDiagonal_v<MT> )
      {
         v_[j+j*mm_] -= (~rhs)(j,j);
      }
      else
      {
         const size_t ibegin( ( IsLower_v<MT> )
                              ?( IsStrictlyLower_v<MT> ? j+1UL : j )
                              :( 0UL ) );
         const size_t iend  ( ( IsUpper_v<MT> )
                              ?( IsStrictlyUpper_v<MT> ? j : j+1UL )
                              :( m_ ) );
         BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

         size_t i( ibegin );

         for( ; (i+2UL) <= iend; i+=2UL ) {
            v_[i  +j*mm_] -= (~rhs)(i  ,j);
            v_[i+1+j*mm_] -= (~rhs)(i+1,j);
         }
         if( i < iend ) {
            v_[i+j*mm_] -= (~rhs)(i,j);
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline void CUDADynamicMatrix<Type,true>::subAssign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block )
      {
         if( IsLower_v<MT> && ii < jj ) continue;
         if( IsUpper_v<MT> && ii > jj ) break;

         for( size_t j=jj; j<jend; ++j )
         {
            const size_t ibegin( ( IsLower_v<MT> )
                                 ?( max( ( IsStrictlyLower_v<MT> ? j+1UL : j ), ii ) )
                                 :( ii ) );
            const size_t iend  ( ( IsUpper_v<MT> )
                                 ?( min( ( IsStrictlyUpper_v<MT> ? j : j+1UL ), m_, ii+block ) )
                                 :( min( m_, ii+block ) ) );
            BLAZE_INTERNAL_ASSERT( ibegin <= iend, "Invalid loop indices detected" );

            for( size_t i=ibegin; i<iend; ++i ) {
               v_[i+j*mm_] -= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::subAssign( const SparseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<(~rhs).columns(); ++j )
      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element )
         v_[element->index()+j*mm_] -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the subtraction assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::subAssign( const SparseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<(~rhs).rows(); ++i )
      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element )
         v_[i+element->index()*mm_] -= element->value();
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a column-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline auto CUDADynamicMatrix<Type,true>::schurAssign( const DenseMatrix<MT,true>& rhs )
{
   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   const size_t ipos( m_ & size_t(-2) );
   BLAZE_INTERNAL_ASSERT( ( m_ - ( m_ % 2UL ) ) == ipos, "Invalid end calculation" );

   for( size_t j=0UL; j<n_; ++j ) {
      for( size_t i=0UL; (i+2UL) <= ipos; i+=2UL ) {
         v_[i  +j*mm_] *= (~rhs)(i  ,j);
         v_[i+1+j*mm_] *= (~rhs)(i+1,j);
      }
      if( ipos < m_ ) {
         v_[ipos+j*mm_] *= (~rhs)(ipos,j);
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a row-major dense matrix.
//
// \param rhs The right-hand side dense matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side dense matrix
inline void CUDADynamicMatrix<Type,true>::schurAssign( const DenseMatrix<MT,false>& rhs )
{
   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   constexpr size_t block( BLOCK_SIZE );

   for( size_t jj=0UL; jj<n_; jj+=block ) {
      const size_t jend( min( n_, jj+block ) );
      for( size_t ii=0UL; ii<m_; ii+=block ) {
         const size_t iend( min( m_, ii+block ) );
         for( size_t j=jj; j<jend; ++j ) {
            for( size_t i=ii; i<iend; ++i ) {
               v_[i+j*mm_] *= (~rhs)(i,j);
            }
         }
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a column-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::schurAssign( const SparseMatrix<MT,true>& rhs )
{
   using blaze::reset;

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t j=0UL; j<n_; ++j )
   {
      size_t i( 0UL );

      for( auto element=(~rhs).begin(j); element!=(~rhs).end(j); ++element ) {
         for( ; i<element->index(); ++i )
            reset( v_[i+j*mm_] );
         v_[i+j*mm_] *= element->value();
         ++i;
      }

      for( ; i<m_; ++i ) {
         reset( v_[i+j*mm_] );
      }
   }
}
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Default implementation of the Schur product assignment of a row-major sparse matrix.
//
// \param rhs The right-hand side sparse matrix for the Schur product.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type >  // Data type of the matrix
template< typename MT >    // Type of the right-hand side sparse matrix
inline void CUDADynamicMatrix<Type,true>::schurAssign( const SparseMatrix<MT,false>& rhs )
{
   using blaze::reset;

   BLAZE_CONSTRAINT_MUST_NOT_BE_SYMMETRIC_MATRIX_TYPE( MT );

   BLAZE_INTERNAL_ASSERT( m_ == (~rhs).rows()   , "Invalid number of rows"    );
   BLAZE_INTERNAL_ASSERT( n_ == (~rhs).columns(), "Invalid number of columns" );

   for( size_t i=0UL; i<m_; ++i )
   {
      size_t j( 0UL );

      for( auto element=(~rhs).begin(i); element!=(~rhs).end(i); ++element ) {
         for( ; j<element->index(); ++j )
            reset( v_[i+j*mm_] );
         v_[i+j*mm_] *= element->value();
         ++j;
      }

      for( ; j<n_; ++j ) {
         reset( v_[i+j*mm_] );
      }
   }
}
/*! \endcond */
//*************************************************************************************************








//=================================================================================================
//
//  DYNAMICMATRIX OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CUDADynamicMatrix operators */
//@{
template< typename Type, bool SO >
void reset( CUDADynamicMatrix<Type,SO>& m );

template< typename Type, bool SO >
void reset( CUDADynamicMatrix<Type,SO>& m, size_t i );

template< typename Type, bool SO >
void clear( CUDADynamicMatrix<Type,SO>& m );

template< bool RF, typename Type, bool SO >
bool isDefault( const CUDADynamicMatrix<Type,SO>& m );

template< typename Type, bool SO >
bool isIntact( const CUDADynamicMatrix<Type,SO>& m ) noexcept;

template< typename Type, bool SO >
void swap( CUDADynamicMatrix<Type,SO>& a, CUDADynamicMatrix<Type,SO>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given dynamic matrix.
// \ingroup dynamic_matrix
//
// \param m The matrix to be resetted.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void reset( CUDADynamicMatrix<Type,SO>& m )
{
   m.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset the specified row/column of the given dynamic matrix.
// \ingroup dynamic_matrix
//
// \param m The matrix to be resetted.
// \param i The index of the row/column to be resetted.
// \return void
//
// This function resets the values in the specified row/column of the given dynamic matrix to
// their default value. In case the given matrix is a \a rowMajor matrix the function resets the
// values in row \a i, if it is a \a columnMajor matrix the function resets the values in column
// \a i. Note that the capacity of the row/column remains unchanged.
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void reset( CUDADynamicMatrix<Type,SO>& m, size_t i )
{
   m.reset( i );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given dynamic matrix.
// \ingroup dynamic_matrix
//
// \param m The matrix to be cleared.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void clear( CUDADynamicMatrix<Type,SO>& m )
{
   m.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given dynamic matrix is in default state.
// \ingroup dynamic_matrix
//
// \param m The matrix to be tested for its default state.
// \return \a true in case the given matrix's rows and columns are zero, \a false otherwise.
//
// This function checks whether the dynamic matrix is in default (constructed) state, i.e. if
// it's number of rows and columns is 0. In case it is in default state, the function returns
// \a true, else it will return \a false. The following example demonstrates the use of the
// \a isDefault() function:

   \code
   blaze::CUDADynamicMatrix<int> A;
   // ... Resizing and initialization
   if( isDefault( A ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (blaze::strict) and relaxed
// semantics (blaze::relaxed):

   \code
   if( isDefault<relaxed>( A ) ) { ... }
   \endcode
*/
template< bool RF        // Relaxation flag
        , typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline bool isDefault( const CUDADynamicMatrix<Type,SO>& m )
{
   return ( m.rows() == 0UL && m.columns() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given dynamic matrix are intact.
// \ingroup dynamic_matrix
//
// \param m The dynamic matrix to be tested.
// \return \a true in case the given matrix's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic matrix are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   blaze::CUDADynamicMatrix<int> A;
   // ... Resizing and initialization
   if( isIntact( A ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline bool isIntact( const CUDADynamicMatrix<Type,SO>& m ) noexcept
{
   return m.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two dynamic matrices.
// \ingroup dynamic_matrix
//
// \param a The first matrix to be swapped.
// \param b The second matrix to be swapped.
// \return void
*/
template< typename Type  // Data type of the matrix
        , bool SO >      // Storage order
inline void swap( CUDADynamicMatrix<Type,SO>& a, CUDADynamicMatrix<Type,SO>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************




// CUDADynamicVector forward declaration
template< typename Type, bool TF > class CUDADynamicVector;




//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct HasConstDataAccess< CUDADynamicMatrix<T,SO> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HASMUTABLEDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct HasMutableDataAccess< CUDADynamicMatrix<T,SO> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISALIGNED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct IsAligned< CUDADynamicMatrix<T,SO> >
   : public BoolConstant<usePadding>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISCONTIGUOUS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct IsContiguous< CUDADynamicMatrix<T,SO> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISPADDED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct IsPadded< CUDADynamicMatrix<T,SO> >
   : public BoolConstant<usePadding>
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISRESIZABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct IsResizable< CUDADynamicMatrix<T,SO> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ISSHRINKABLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct IsShrinkable< CUDADynamicMatrix<T,SO> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ADDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, bool SO1, bool SO2 >
struct AddTrait< CUDADynamicMatrix<T1, SO1>, CUDADynamicMatrix<T2, SO2> >
{
   static constexpr bool SO = ( IsDenseMatrix_v<T1> && IsDenseMatrix_v<T2>
                                ? ( IsSymmetric_v<T1> ^ IsSymmetric_v<T2>
                                    ? ( IsSymmetric_v<T1>
                                        ? SO2
                                        : SO1 )
                                    : SO1 && SO2 )
                                : ( IsDenseMatrix_v<T1>
                                    ? SO1
                                    : SO2 ) );

   using Type = CUDADynamicMatrix< AddTrait_t<T1,T2>, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, bool SO1, bool SO2 >
struct SubTrait< CUDADynamicMatrix<T1, SO1>, CUDADynamicMatrix<T2, SO2> >
{
   static constexpr bool SO = ( IsDenseMatrix_v<T1> && IsDenseMatrix_v<T2>
                                ? ( IsSymmetric_v<T1> ^ IsSymmetric_v<T2>
                                    ? ( IsSymmetric_v<T1>
                                        ? SO2
                                        : SO1 )
                                    : SO1 && SO2 )
                                : ( IsDenseMatrix_v<T1>
                                    ? SO1
                                    : SO2 ) );

   using Type = CUDADynamicMatrix< SubTrait_t<T1,T2>, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SCHURTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, bool SO1, bool SO2 >
struct SchurTrait< CUDADynamicMatrix<T1, SO1>, CUDADynamicMatrix<T2, SO2> >
{
   static constexpr bool SO = ( IsSymmetric_v<T1> ^ IsSymmetric_v<T2>
                                ? ( IsSymmetric_v<T1>
                                    ? SO2
                                    : SO1 )
                                : SO1 && SO2 );

   using Type = CUDADynamicMatrix< MultTrait_t<T1,T2>, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MULTTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, bool SO >
struct MultTrait< CUDADynamicMatrix<T1, SO>, T2, EnableIf_t< IsNumeric_v<T2> > >
{
   using Type = CUDADynamicMatrix< MultTrait_t<T1,T2>, SO >;
};

template< typename T1, typename T2, bool SO >
struct MultTrait< T1, CUDADynamicMatrix<T2, SO>, EnableIf_t< IsNumeric_v<T1> > >
{
   using Type = CUDADynamicMatrix< MultTrait_t<T1,T2>, SO >;
};

template< typename T1, typename T2 >
struct MultTrait< CUDADynamicVector<T1, columnVector>, CUDADynamicVector<T2, rowVector> >
{
   using Type = CUDADynamicMatrix< MultTrait_t<T1,T2>, false >;
};

template< typename T1, typename T2, bool SO1, bool SO2 >
struct MultTrait< CUDADynamicMatrix<T1, SO1>, CUDADynamicMatrix<T2, SO2> >
{
   static constexpr bool SO = ( IsSparseMatrix_v<T1> ? SO2 : SO1 );

   using Type = CUDADynamicMatrix< MultTrait_t<T1,T2>, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DIVTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, bool SO1, bool SO2 >
struct DivTrait< CUDADynamicMatrix<T1, SO1>, CUDADynamicMatrix<T2, SO2> >
{
   using Type = CUDADynamicMatrix< DivTrait_t<T1,T2>, SO1 >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  MAPTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO, typename OP >
struct MapTrait< CUDADynamicMatrix<T, SO>, OP >
{
   using Type = CUDADynamicMatrix< MapTrait_t<T,OP>, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  EXPANDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF, size_t... CEAs >
struct ExpandTrait< CUDADynamicVector<T,TF>, CEAs... >
{
   using Type = CUDADynamicMatrix<T,( TF == columnVector ? columnMajor : rowMajor )>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  HIGHTYPE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, bool SO, typename T2 >
struct HighType< CUDADynamicMatrix<T1,SO>, CUDADynamicMatrix<T2,SO> >
{
   using Type = CUDADynamicMatrix< typename HighType<T1,T2>::Type, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LOWTYPE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, bool SO, typename T2 >
struct LowType< CUDADynamicMatrix<T1,SO>, CUDADynamicMatrix<T2,SO> >
{
   using Type = CUDADynamicMatrix< typename LowType<T1,T2>::Type, SO >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBMATRIXTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO >
struct SubmatrixTrait< CUDADynamicMatrix<T,SO> >
{
   using Type = CUDADynamicMatrix<T,SO>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWSTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO, size_t M >
struct RowsTrait< CUDADynamicMatrix<T,SO>, M >
{
   using Type = CUDADynamicMatrix<T,false>;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNSTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool SO, size_t N >
struct ColumnsTrait< CUDADynamicMatrix<T,SO>, N >
{
   using Type = CUDADynamicMatrix<T,true>;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif
