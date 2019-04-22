//=================================================================================================
/*!
//  \file blaze/math/dense/CUDAManagedVector.h
//  \brief Header file for the implementation of an arbitrarily sized vector
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

#ifndef _BLAZE_CUDA_MATH_DENSE_CUDAMANAGEDVECTOR_H_
#define _BLAZE_CUDA_MATH_DENSE_CUDAMANAGEDVECTOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <utility>
#include <blaze/math/Aliases.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/constraints/DenseVector.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/TransposeFlag.h>
#include <blaze/math/dense/DenseIterator.h>
#include <blaze/math/Exception.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/expressions/SparseVector.h>
#include <blaze/math/Forward.h>
#include <blaze/math/InitializerList.h>
#include <blaze/math/shims/Clear.h>
#include <blaze/math/shims/IsDefault.h>
#include <blaze/math/shims/NextMultiple.h>
#include <blaze/math/shims/Serial.h>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/traits/BandTrait.h>
#include <blaze/math/traits/ColumnTrait.h>
#include <blaze/math/traits/CrossTrait.h>
#include <blaze/math/traits/DivTrait.h>
#include <blaze/math/traits/ElementsTrait.h>
#include <blaze/math/traits/MapTrait.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/traits/ReduceTrait.h>
#include <blaze/math/traits/RowTrait.h>
#include <blaze/math/traits/SubTrait.h>
#include <blaze/math/traits/SubvectorTrait.h>
#include <blaze/math/typetraits/HasConstDataAccess.h>
#include <blaze/math/typetraits/HasMutableDataAccess.h>
#include <blaze/math/typetraits/HighType.h>
#include <blaze/math/typetraits/IsAligned.h>
#include <blaze/math/typetraits/IsColumnVector.h>
#include <blaze/math/typetraits/IsContiguous.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsDenseVector.h>
#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/math/typetraits/IsResizable.h>
#include <blaze/math/typetraits/IsRowVector.h>
#include <blaze/math/typetraits/IsShrinkable.h>
#include <blaze/math/typetraits/IsSMPAssignable.h>
#include <blaze/math/typetraits/IsSparseVector.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/math/typetraits/LowType.h>
#include <blaze/math/typetraits/MaxSize.h>
#include <blaze/math/typetraits/Size.h>
#include <blaze/math/typetraits/TransposeFlag.h>
#include <blaze/system/CacheSize.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Restrict.h>
#include <blaze/system/Thresholds.h>
#include <blaze/system/TransposeFlag.h>
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
#include <blaze/util/Memory.h>
#include <blaze/util/TrueType.h>
#include <blaze/util/Types.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsVectorizable.h>
#include <blaze/util/typetraits/RemoveConst.h>

#include <blaze_cuda/math/typetraits/IsCUDAEnabled.h>
#include <blaze_cuda/util/algorithms/CUDATransform.h>

#include <cuda_runtime.h>

namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\defgroup dynamic_vector CUDAManagedVector
// \ingroup dense_vector
*/
/*!\brief Efficient implementation of an arbitrary sized vector.
// \ingroup dynamic_vector
//
// The CUDAManagedVector class template is the representation of an arbitrary sized vector with
// dynamically allocated elements of arbitrary type. The type of the elements and the transpose
// flag of the vector can be specified via the two template parameters:

   \code
   template< typename Type, bool TF >
   class CUDAManagedVector;
   \endcode

//  - Type: specifies the type of the vector elements. CUDAManagedVector can be used with any
//          non-cv-qualified, non-reference, non-pointer element type.
//  - TF  : specifies whether the vector is a row vector (\a rowVector) or a column
//          vector (\a columnVector). The default value is \a columnVector.
//
// These contiguously stored elements can be directly accessed with the subscript operator. The
// numbering of the vector elements is

                             \f[\left(\begin{array}{*{5}{c}}
                             0 & 1 & 2 & \cdots & N-1 \\
                             \end{array}\right)\f]

// The use of CUDAManagedVector is very natural and intuitive. All operations (addition, subtraction,
// multiplication, scaling, ...) can be performed on all possible combinations of dense and sparse
// vectors with fitting element types. The following example gives an impression of the use of
// CUDAManagedVector:

   \code
   using blaze::CUDAManagedVector;
   using blaze::CompressedVector;
   using blaze::DynamicMatrix;

   CUDAManagedVector<double> a( 2 );  // Non-initialized 2D vector of size 2
   a[0] = 1.0;                    // Initialization of the first element
   a[1] = 2.0;                    // Initialization of the second element

   CUDAManagedVector<double>   b( 2, 2.0  );  // Directly, homogeneously initialized 2D vector
   CompressedVector<float> c( 2 );        // Empty sparse single precision vector
   CUDAManagedVector<double>   d;             // Default constructed dynamic vector
   DynamicMatrix<double>   A;             // Default constructed row-major matrix

   d = a + b;  // Vector addition between vectors of equal element type
   d = a - c;  // Vector subtraction between a dense and sparse vector with different element types
   d = a * b;  // Component-wise vector multiplication

   a *= 2.0;      // In-place scaling of vector
   d  = a * 2.0;  // Scaling of vector a
   d  = 2.0 * a;  // Scaling of vector a

   d += a - b;  // Addition assignment
   d -= a + c;  // Subtraction assignment
   d *= a * b;  // Multiplication assignment

   double scalar = trans( a ) * b;  // Scalar/dot/inner product between two vectors

   A = a * trans( b );  // Outer product between two vectors
   \endcode
*/
template< typename Type                     // Data type of the vector
        , bool TF = defaultTransposeFlag >  // Transpose flag
class CUDAManagedVector
   //: public DenseVector< CUDAManagedVector<Type,TF>, TF >
{
 public:
   //**Type definitions****************************************************************************
   using This          = CUDAManagedVector<Type,TF>;    //!< Type of this CUDAManagedVector instance.
   using BaseType      = DenseVector<This,TF>;      //!< Base type of this CUDAManagedVector instance.
   using ResultType    = This;                      //!< Result type for expression template evaluations.
   using TransposeType = CUDAManagedVector<Type,!TF>;   //!< Transpose type for expression template evaluations.
   using ElementType   = Type;                      //!< Type of the vector elements.
   using ReturnType    = const Type&;               //!< Return type for expression template evaluations
   using CompositeType = const CUDAManagedVector&;      //!< Data type for composite expression templates.

   using Reference      = Type&;        //!< Reference to a non-constant vector value.
   using ConstReference = const Type&;  //!< Reference to a constant vector value.
   using Pointer        = Type*;        //!< Pointer to a non-constant vector value.
   using ConstPointer   = const Type*;  //!< Pointer to a constant vector value.

   using Iterator      = DenseIterator<Type,aligned>;        //!< Iterator over non-constant elements.
   using ConstIterator = DenseIterator<const Type,aligned>;  //!< Iterator over constant elements.
   //**********************************************************************************************

   //**Rebind struct definition********************************************************************
   /*!\brief Rebind mechanism to obtain a CUDAManagedVector with different data/element type.
   */
   template< typename NewType >  // Data type of the other vector
   struct Rebind {
      using Other = CUDAManagedVector<NewType,TF>;  //!< The type of the other CUDAManagedVector.
   };
   //**********************************************************************************************

   //**Resize struct definition********************************************************************
   /*!\brief Resize mechanism to obtain a CUDAManagedVector with a different fixed number of elements.
   */
   template< size_t NewN >  // Number of elements of the other vector
   struct Resize {
      using Other = CUDAManagedVector<Type,TF>;  //!< The type of the other CUDAManagedVector.
   };
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline CUDAManagedVector() noexcept;
   explicit inline CUDAManagedVector( size_t n );
   explicit inline CUDAManagedVector( size_t n, const Type& init );
   explicit inline CUDAManagedVector( std::initializer_list<Type> list );

   template< typename Other >
   explicit inline CUDAManagedVector( size_t n, const Other* array );

   template< typename Other, size_t Dim >
   explicit inline CUDAManagedVector( const Other (&array)[Dim] );

                           inline CUDAManagedVector( const CUDAManagedVector& v );
                           inline CUDAManagedVector( CUDAManagedVector&& v ) noexcept;
   template< typename VT > inline CUDAManagedVector( const Vector<VT,TF>& v );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   /*!\name Destructor */
   //@{
   inline ~CUDAManagedVector();
   //@}
   //**********************************************************************************************

   //**Data access functions***********************************************************************
   /*!\name Data access functions */
   //@{
   inline __host__ __device__ Reference              operator[]( size_t index ) noexcept;
   inline __host__ __device__ ConstReference         operator[]( size_t index ) const noexcept;
   inline __host__ __device__ Reference              at( size_t index );
   inline __host__ __device__ ConstReference         at( size_t index ) const;
   inline __host__ __device__ Pointer                data  () noexcept;
   inline __host__ __device__ ConstPointer           data  () const noexcept;
   inline __host__ __device__ Iterator        begin () noexcept;
   inline __host__ __device__ ConstIterator   begin () const noexcept;
   inline __host__ __device__ ConstIterator   cbegin() const noexcept;
   inline __host__ __device__ Iterator        end   () noexcept;
   inline __host__ __device__ ConstIterator   end   () const noexcept;
   inline __host__ __device__ ConstIterator   cend  () const noexcept;
   //@}
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   inline CUDAManagedVector& operator=( const Type& rhs );
   inline CUDAManagedVector& operator=( std::initializer_list<Type> list );

   template< typename Other, size_t Dim >
   inline CUDAManagedVector& operator=( const Other (&array)[Dim] );

   inline CUDAManagedVector& operator=( const CUDAManagedVector& rhs );
   inline CUDAManagedVector& operator=( CUDAManagedVector&& rhs ) noexcept;

   template< typename VT > inline CUDAManagedVector& operator= ( const Vector<VT,TF>& rhs );
   template< typename VT > inline CUDAManagedVector& operator+=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CUDAManagedVector& operator-=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CUDAManagedVector& operator*=( const Vector<VT,TF>& rhs );
   template< typename VT > inline CUDAManagedVector& operator/=( const DenseVector<VT,TF>& rhs );
   template< typename VT > inline CUDAManagedVector& operator%=( const Vector<VT,TF>& rhs );
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t __host__ __device__ size() const noexcept;
   inline size_t __host__ __device__ spacing() const noexcept;
   inline size_t __host__ __device__ capacity() const noexcept;
   inline size_t nonZeros() const;
   inline void   reset();
   inline void   clear();
   inline void   resize( size_t n, bool preserve=true );
   inline void   extend( size_t n, bool preserve=true );
   inline void   reserve( size_t n );
   inline void   shrinkToFit();
   inline void   swap( CUDAManagedVector& v ) noexcept;
   //@}
   //**********************************************************************************************

   //**Numeric functions***************************************************************************
   /*!\name Numeric functions */
   //@{
   template< typename Other > inline CUDAManagedVector& scale( const Other& scalar );
   //@}
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

   inline bool isAligned   () const noexcept;
   inline bool canSMPAssign() const noexcept;

   template< typename VT >  // Type of the right-hand side dense vector
   inline auto assign( const DenseVector<VT,TF>& rhs ) -> EnableIf_t< IsCUDAEnabled_v<VT> >;
   template< typename VT >  // Type of the right-hand side dense vector
   inline auto assign( const DenseVector<VT,TF>& rhs ) -> DisableIf_t< IsCUDAEnabled_v<VT> >;

   template< typename VT > inline void assign( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline void addAssign( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline void subAssign( const SparseVector<VT,TF>& rhs );
   template< typename VT > inline void multAssign( const SparseVector<VT,TF>& rhs );
   //@}
   //**********************************************************************************************

 private:
   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline size_t addPadding( size_t value ) const noexcept;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t size_;             //!< The current size/dimension of the vector.
   size_t capacity_;         //!< The maximum capacity of the vector.
   Type* BLAZE_RESTRICT v_;  //!< The dynamically allocated vector elements.
                             /*!< Access to the vector elements is gained via the subscript operator.
                                  The order of the elements is
                                  \f[\left(\begin{array}{*{5}{c}}
                                  0 & 1 & 2 & \cdots & N-1 \\
                                  \end{array}\right)\f] */
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
/*!\brief The default constructor for CUDAManagedVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::CUDAManagedVector() noexcept
   : size_    ( 0UL )      // The current size/dimension of the vector
   , capacity_( 0UL )      // The maximum capacity of the vector
   , v_       ( nullptr )  // The vector elements
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a vector of size \a n. No element initialization is performed!
//
// \param n The size of the vector.
//
// \note This constructor is only responsible to allocate the required dynamic memory. No
// element initialization is performed!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( size_t n )
   : size_    ( n )                            // The current size/dimension of the vector
   , capacity_( n )              // The maximum capacity of the vector
   //, v_       ( allocate<Type>( capacity_ ) )  // The vector elements
{
   cudaMallocManaged((void**)v_, n * sizeof(Type));   // TODO: Make an allocator, maybe..?
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for a homogeneous initialization of all \a n vector elements.
//
// \param n The size of the vector.
// \param init The initial value of the vector elements.
//
// All vector elements are initialized with the specified value.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( size_t n, const Type& init )
   : CUDAManagedVector( n )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] = init;

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List initialization of all vector elements.
//
// \param list The initializer list.
//
// This constructor provides the option to explicitly initialize the elements of the vector
// within a constructor call:

   \code
   CUDAManagedVector<double> v1{ 4.2, 6.3, -1.2 };
   \endcode

// The vector is sized according to the size of the initializer list and all its elements are
// (copy) assigned the elements of the given initializer list.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( std::initializer_list<Type> list )
   : CUDAManagedVector( list.size() )
{
   std::fill( std::copy( list.begin(), list.end(), begin() ), end(), Type() );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all vector elements.
//
// \param n The size of the vector.
// \param array Dynamic array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the vector with a
// dynamic array:

   \code
   double* array = new double[4];
   // ... Initialization of the dynamic array
   CUDAManagedVector<double> v( array, 4UL );
   delete[] array;
   \endcode

// The vector is sized according to the specified size of the array and initialized with the
// values from the given array. Note that it is expected that the given \a array has at least
// \a n elements. Providing an array with less elements results in undefined behavior!
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the initialization array
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( size_t n, const Other* array )
   : CUDAManagedVector( n )
{
   for( size_t i=0UL; i<n; ++i )
      v_[i] = array[i];

   //BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array initialization of all vector elements.
//
// \param array N-dimensional array for the initialization.
//
// This constructor offers the option to directly initialize the elements of the vector with a
// static array:

   \code
   const int init[4] = { 1, 2, 3 };
   CUDAManagedVector<int> v( init );
   \endcode

// The vector is sized according to the size of the array and initialized with the values from the
// given array. Missing values are initialized with default values (as e.g. the fourth element in
// the example).
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Other  // Data type of the initialization array
        , size_t Dim >    // Dimension of the initialization array
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( const Other (&array)[Dim] )
   : CUDAManagedVector( Dim )
{
   for( size_t i=0UL; i<Dim; ++i )
      v_[i] = array[i];

   //BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The copy constructor for CUDAManagedVector.
//
// \param v Vector to be copied.
//
// The copy constructor is explicitly defined due to the required dynamic memory management
// and in order to enable/facilitate NRV optimization.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( const CUDAManagedVector& v )
   : CUDAManagedVector( v.size_ )
{
   BLAZE_INTERNAL_ASSERT( capacity_ <= v.capacity_, "Invalid capacity estimation" );

   cudaAssign( *this, ~v );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief The move constructor for CUDAManagedVector.
//
// \param v The vector to be moved into this instance.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( CUDAManagedVector&& v ) noexcept
   : size_    ( v.size_     )  // The current size/dimension of the vector
   , capacity_( v.capacity_ )  // The maximum capacity of the vector
   , v_       ( v.v_        )  // The vector elements
{
   v.size_     = 0UL;
   v.capacity_ = 0UL;
   v.v_        = nullptr;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different vectors.
//
// \param v Vector to be copied.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the foreign vector
inline CUDAManagedVector<Type,TF>::CUDAManagedVector( const Vector<VT,TF>& v )
   : CUDAManagedVector( (~v).size() )
{
   if( IsSparseVector_v<VT> ) {
      for( size_t i=0UL; i<size_; ++i ) {
         v_[i] = Type();
      }
   }

   smpAssign( *this, ~v );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );
}
//*************************************************************************************************




//=================================================================================================
//
//  DESTRUCTOR
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The destructor for CUDAManagedVector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>::~CUDAManagedVector()
{
   cudaFree(v_);
}
//*************************************************************************************************




//=================================================================================================
//
//  DATA ACCESS FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::Reference
   CUDAManagedVector<Type,TF>::operator[]( size_t index ) noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid vector access index" );
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subscript operator for the direct access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
//
// This function only performs an index check in case BLAZE_USER_ASSERT() is active. In contrast,
// the at() function is guaranteed to perform a check of the given access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstReference
   CUDAManagedVector<Type,TF>::operator[]( size_t index ) const noexcept
{
   BLAZE_USER_ASSERT( index < size_, "Invalid vector access index" );
   return v_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid vector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::Reference
   CUDAManagedVector<Type,TF>::at( size_t index )
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
   }
   return (*this)[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checked access to the vector elements.
//
// \param index Access index. The index has to be in the range \f$[0..N-1]\f$.
// \return Reference to the accessed value.
// \exception std::out_of_range Invalid vector access index.
//
// In contrast to the subscript operator this function always performs a check of the given
// access index.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstReference
   CUDAManagedVector<Type,TF>::at( size_t index ) const
{
   if( index >= size_ ) {
      BLAZE_THROW_OUT_OF_RANGE( "Invalid vector access index" );
   }
   return (*this)[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::Pointer CUDAManagedVector<Type,TF>::data() noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Low-level data access to the vector elements.
//
// \return Pointer to the internal element storage.
//
// This function returns a pointer to the internal storage of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstPointer CUDAManagedVector<Type,TF>::data() const noexcept
{
   return v_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the dynamic vector.
//
// \return Iterator to the first element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::Iterator CUDAManagedVector<Type,TF>::begin() noexcept
{
   return Iterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the dynamic vector.
//
// \return Iterator to the first element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstIterator CUDAManagedVector<Type,TF>::begin() const noexcept
{
   return ConstIterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator to the first element of the dynamic vector.
//
// \return Iterator to the first element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstIterator CUDAManagedVector<Type,TF>::cbegin() const noexcept
{
   return ConstIterator( v_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the dynamic vector.
//
// \return Iterator just past the last element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::Iterator CUDAManagedVector<Type,TF>::end() noexcept
{
   return Iterator( v_ + size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the dynamic vector.
//
// \return Iterator just past the last element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstIterator CUDAManagedVector<Type,TF>::end() const noexcept
{
   return ConstIterator( v_ + size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns an iterator just past the last element of the dynamic vector.
//
// \return Iterator just past the last element of the dynamic vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ typename CUDAManagedVector<Type,TF>::ConstIterator CUDAManagedVector<Type,TF>::cend() const noexcept
{
   return ConstIterator( v_ + size_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Homogenous assignment to all vector elements.
//
// \param rhs Scalar value to be assigned to all vector elements.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator=( const Type& rhs )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] = rhs;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief List assignment to all vector elements.
//
// \param list The initializer list.
//
// This assignment operator offers the option to directly assign to all elements of the vector
// by means of an initializer list:

   \code
   CUDAManagedVector<double> v;
   v = { 4.2, 6.3, -1.2 };
   \endcode

// The vector is resized according to the size of the initializer list and all its elements are
// (copy) assigned the values from the given initializer list.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator=( std::initializer_list<Type> list )
{
   resize( list.size(), false );
   std::copy( list.begin(), list.end(), v_ );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Array assignment to all vector elements.
//
// \param array N-dimensional array for the assignment.
// \return Reference to the assigned vector.
//
// This assignment operator offers the option to directly set all elements of the vector:

   \code
   const int init[4] = { 1, 2, 3 };
   CUDAManagedVector<int> v;
   v = init;
   \endcode

// The vector is resized according to the size of the array and assigned the values from the given
// array. Missing values are initialized with default values (as e.g. the fourth element in the
// example).
*/
template< typename Type   // Data type of the vector
        , bool TF >       // Transpose flag
template< typename Other  // Data type of the initialization array
        , size_t Dim >    // Dimension of the initialization array
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator=( const Other (&array)[Dim] )
{
   resize( Dim, false );

   for( size_t i=0UL; i<Dim; ++i )
      v_[i] = array[i];

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Copy assignment operator for CUDAManagedVector.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
//
// The vector is resized according to the given N-dimensional vector and initialized as a
// copy of this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator=( const CUDAManagedVector& rhs )
{
   if( &rhs == this ) return *this;

   resize( rhs.size_, false );
   smpAssign( *this, ~rhs );

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Move assignment operator for CUDAManagedVector.
//
// \param rhs The vector to be moved into this instance.
// \return Reference to the assigned vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator=( CUDAManagedVector&& rhs ) noexcept
{
   cudaFree( v_ );

   size_     = rhs.size_;
   capacity_ = rhs.capacity_;
   v_        = rhs.v_;

   rhs.size_     = 0UL;
   rhs.capacity_ = 0UL;
   rhs.v_        = nullptr;

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Assignment operator for different vectors.
//
// \param rhs Vector to be copied.
// \return Reference to the assigned vector.
//
// The vector is resized according to the given vector and initialized as a copy of this vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).canAlias( this ) ) {
      CUDAManagedVector tmp( ~rhs );
      swap( tmp );
   }
   else {
      resize( (~rhs).size(), false );
      if( IsSparseVector_v<VT> )
         reset();
      smpAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition assignment operator for the addition of a vector (\f$ \vec{a}+=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be added to the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator+=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<VT> tmp( ~rhs );
      smpAddAssign( *this, tmp );
   }
   else {
      smpAddAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator for the subtraction of a vector
//        (\f$ \vec{a}-=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be subtracted from the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator-=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      const ResultType_t<VT> tmp( ~rhs );
      smpSubAssign( *this, tmp );
   }
   else {
      smpSubAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Multiplication assignment operator for the multiplication of a vector
//        (\f$ \vec{a}*=\vec{b} \f$).
//
// \param rhs The right-hand side vector to be multiplied with the vector.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator*=( const Vector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( IsSparseVector_v<VT> || (~rhs).canAlias( this ) ) {
      CUDAManagedVector<Type,TF> tmp( *this * (~rhs) );
      swap( tmp );
   }
   else {
      smpMultAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Division assignment operator for the division of a dense vector (\f$ \vec{a}/=\vec{b} \f$).
//
// \param rhs The right-hand side dense vector divisor.
// \return Reference to the vector.
// \exception std::invalid_argument Vector sizes do not match.
//
// In case the current sizes of the two vectors don't match, a \a std::invalid_argument exception
// is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator/=( const DenseVector<VT,TF>& rhs )
{
   if( (~rhs).size() != size_ ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Vector sizes do not match" );
   }

   if( (~rhs).canAlias( this ) ) {
      CUDAManagedVector<Type,TF> tmp( *this / (~rhs) );
      swap( tmp );
   }
   else {
      smpDivAssign( *this, ~rhs );
   }

   BLAZE_INTERNAL_ASSERT( isIntact(), "Invariant violation detected" );

   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Cross product assignment operator for the multiplication of a vector
//        (\f$ \vec{a}\times=\vec{b} \f$).
//
// \param rhs The right-hand side vector for the cross product.
// \return Reference to the vector.
// \exception std::invalid_argument Invalid vector size for cross product.
//
// In case the current size of any of the two vectors is not equal to 3, a \a std::invalid_argument
// exception is thrown.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side vector
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::operator%=( const Vector<VT,TF>& rhs )
{
   //using assign;

   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( ResultType_t<VT>, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ResultType_t<VT> );

   using CrossType = CrossTrait_t< This, ResultType_t<VT> >;

   BLAZE_CONSTRAINT_MUST_BE_DENSE_VECTOR_TYPE( CrossType );
   BLAZE_CONSTRAINT_MUST_BE_VECTOR_WITH_TRANSPOSE_FLAG( CrossType, TF );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( CrossType );

   if( size_ != 3UL || (~rhs).size() != 3UL ) {
      BLAZE_THROW_INVALID_ARGUMENT( "Invalid vector size for cross product" );
   }

   const CrossType tmp( *this % (~rhs) );
   assign( *this, tmp );

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
/*!\brief Returns the current size/dimension of the vector.
//
// \return The size of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ size_t CUDAManagedVector<Type,TF>::size() const noexcept
{
   return size_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the minimum capacity of the vector.
//
// \return The minimum capacity of the vector.
//
// This function returns the minimum capacity of the vector, which corresponds to the current
// size plus padding.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ size_t CUDAManagedVector<Type,TF>::spacing() const noexcept
{
   return addPadding( size_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the maximum capacity of the vector.
//
// \return The maximum capacity of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline __host__ __device__ size_t CUDAManagedVector<Type,TF>::capacity() const noexcept
{
   return capacity_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns the number of non-zero elements in the vector.
//
// \return The number of non-zero elements in the vector.
//
// Note that the number of non-zero elements is always less than or equal to the current size
// of the vector.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline size_t CUDAManagedVector<Type,TF>::nonZeros() const
{
   size_t nonzeros( 0 );

   for( size_t i=0UL; i<size_; ++i ) {
      if( !isDefault( v_[i] ) )
         ++nonzeros;
   }

   return nonzeros;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Reset to the default initial values.
//
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::reset()
{
   //using clear;
   for( size_t i=0UL; i<size_; ++i )
      clear( v_[i] );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the vector.
//
// \return void
//
// After the clear() function, the size of the vector is 0.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::clear()
{
   resize( 0UL, false );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Changing the size of the vector.
//
// \param n The new size of the vector.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function resizes the vector using the given size to \a n. During this operation, new
// dynamic memory may be allocated in case the capacity of the vector is too small. Note that
// this function may invalidate all existing views (subvectors, ...) on the vector if it is
// used to shrink the vector. Additionally, the resize operation potentially changes all vector
// elements. In order to preserve the old vector values, the \a preserve flag can be set to
// \a true. However, new vector elements are not initialized!
//
// The following example illustrates the resize operation of a vector of size 2 to a vector of
// size 4. The new, uninitialized elements are marked with \a x:

                              \f[
                              \left(\begin{array}{*{2}{c}}
                              1 & 2 \\
                              \end{array}\right)

                              \Longrightarrow

                              \left(\begin{array}{*{4}{c}}
                              1 & 2 & x & x \\
                              \end{array}\right)
                              \f]
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::resize( size_t n, bool preserve )
{
   using std::swap;

   if( n > capacity_ )
   {
      // Allocating a new array
      const size_t newCapacity( addPadding( n ) );
      //Type* BLAZE_RESTRICT tmp = allocate<Type>( newCapacity );
      Type* tmp;
      cudaMallocManaged( (void**)&tmp, newCapacity );

      // Initializing the new array
      if( preserve ) {
         transfer( v_, v_+size_, tmp );
      }

      if( IsVectorizable_v<Type> ) {
         for( size_t i=size_; i<newCapacity; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      swap( v_, tmp );
      cudaFree( tmp );
      capacity_ = newCapacity;
   }
   else if( IsVectorizable_v<Type> && n < size_ )
   {
      for( size_t i=n; i<size_; ++i )
         v_[i] = Type();
   }

   size_ = n;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Extending the size of the vector.
//
// \param n Number of additional vector elements.
// \param preserve \a true if the old values of the vector should be preserved, \a false if not.
// \return void
//
// This function increases the vector size by \a n elements. During this operation, new dynamic
// memory may be allocated in case the capacity of the vector is too small. Therefore this
// function potentially changes all vector elements. In order to preserve the old vector values,
// the \a preserve flag can be set to \a true. However, new vector elements are not initialized!
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::extend( size_t n, bool preserve )
{
   resize( size_+n, preserve );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Setting the minimum capacity of the vector.
//
// \param n The new minimum capacity of the vector.
// \return void
//
// This function increases the capacity of the vector to at least \a n elements. The current
// values of the vector elements are preserved.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::reserve( size_t n )
{
   using std::swap;

   if( n > capacity_ )
   {
      // Allocating a new array
      const size_t newCapacity( addPadding( n ) );
      Type* BLAZE_RESTRICT tmp = allocate<Type>( newCapacity );

      // Initializing the new array
      transfer( v_, v_+size_, tmp );

      if( IsVectorizable_v<Type> ) {
         for( size_t i=size_; i<newCapacity; ++i )
            tmp[i] = Type();
      }

      // Replacing the old array
      swap( tmp, v_ );
      cudaFree( tmp );
      capacity_ = newCapacity;
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Requesting the removal of unused capacity.
//
// \return void
//
// This function minimizes the capacity of the vector by removing unused capacity. Please note
// that due to padding the capacity might not be reduced exactly to size(). Please also note
// that in case a reallocation occurs, all iterators (including end() iterators), all pointers
// and references to elements of this vector are invalidated.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::shrinkToFit()
{
   if( spacing() < capacity_ ) {
      CUDAManagedVector( *this ).swap( *this );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two vectors.
//
// \param v The vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void CUDAManagedVector<Type,TF>::swap( CUDAManagedVector& v ) noexcept
{
   using std::swap;

   swap( size_, v.size_ );
   swap( capacity_, v.capacity_ );
   swap( v_, v.v_ );
}
//*************************************************************************************************




//=================================================================================================
//
//  NUMERIC FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Scaling of the vector by the scalar value \a scalar (\f$ \vec{a}=\vec{b}*s \f$).
//
// \param scalar The scalar value for the vector scaling.
// \return Reference to the vector.
//
// This function scales the vector by applying the given scalar value \a scalar to each element
// of the vector. For built-in and \c complex data types it has the same effect as using the
// multiplication assignment operator:

   \code
   CUDAManagedVector<int> a;
   // ... Initialization
   a *= 4;        // Scaling of the vector
   a.scale( 4 );  // Same effect as above
   \endcode
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the scalar value
inline CUDAManagedVector<Type,TF>& CUDAManagedVector<Type,TF>::scale( const Other& scalar )
{
   for( size_t i=0UL; i<size_; ++i )
      v_[i] *= scalar;
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  DEBUGGING FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns whether the invariants of the dynamic vector are intact.
//
// \return \a true in case the dynamic vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool CUDAManagedVector<Type,TF>::isIntact() const noexcept
{
   if( size_ > capacity_ )
      return false;

   if( IsVectorizable_v<Type> ) {
      for( size_t i=size_; i<capacity_; ++i ) {
         if( v_[i] != Type() )
            return false;
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
/*!\brief Returns whether the vector can alias with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this vector, \a false if not.
//
// This function returns whether the given address can alias with the vector. In contrast
// to the isAliased() function this function is allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CUDAManagedVector<Type,TF>::canAlias( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the vector is aliased with the given address \a alias.
//
// \param alias The alias to be checked.
// \return \a true in case the alias corresponds to this vector, \a false if not.
//
// This function returns whether the given address is aliased with the vector. In contrast
// to the canAlias() function this function is not allowed to use compile time expressions
// to optimize the evaluation.
*/
template< typename Type     // Data type of the vector
        , bool TF >         // Transpose flag
template< typename Other >  // Data type of the foreign expression
inline bool CUDAManagedVector<Type,TF>::isAliased( const Other* alias ) const noexcept
{
   return static_cast<const void*>( this ) == static_cast<const void*>( alias );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the vector is properly aligned in memory.
//
// \return \a true in case the vector is aligned, \a false if not.
//
// This function returns whether the vector is guaranteed to be properly aligned in memory, i.e.
// whether the beginning and the end of the vector are guaranteed to conform to the alignment
// restrictions of the element type \a Type.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool CUDAManagedVector<Type,TF>::isAligned() const noexcept
{
   return true;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the vector can be used in SMP assignments.
//
// \return \a true in case the vector can be used in SMP assignments, \a false if not.
//
// This function returns whether the vector can be used in SMP assignments. In contrast to the
// \a smpAssignable member enumeration, which is based solely on compile time information, this
// function additionally provides runtime information (as for instance the current size of the
// vector).
*/
//template< typename Type  // Data type of the vector
//        , bool TF >      // Transpose flag
//inline bool CUDAManagedVector<Type,TF>::canSMPAssign() const noexcept
//{
//   return ( size() > SMP_DVECASSIGN_THRESHOLD );
//}
//*************************************************************************************************


template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline auto CUDAManagedVector<Type,TF>::assign( const DenseVector<VT,TF>& rhs )
   -> EnableIf_t< IsCUDAEnabled_v<VT> >
{
   cuda_copy ( rhs.begin(), rhs.end(), begin() );
}

template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side dense vector
inline auto CUDAManagedVector<Type,TF>::assign( const DenseVector<VT,TF>& rhs )
   -> DisableIf_t< IsCUDAEnabled_v<VT> >
{
   using std::copy;
   copy( rhs.begin(), rhs.end(), begin() );
}


//*************************************************************************************************
/*!\brief Default implementation of the assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be assigned.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CUDAManagedVector<Type,TF>::assign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( auto element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the addition assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be added.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CUDAManagedVector<Type,TF>::addAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( auto element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] += element->value();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Default implementation of the subtraction assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be subtracted.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CUDAManagedVector<Type,TF>::subAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   for( auto element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] -= element->value();
}
//*************************************************************************************************



//*************************************************************************************************
/*!\brief Default implementation of the multiplication assignment of a sparse vector.
//
// \param rhs The right-hand side sparse vector to be multiplied.
// \return void
//
// This function must \b NOT be called explicitly! It is used internally for the performance
// optimized evaluation of expression templates. Calling this function explicitly might result
// in erroneous results and/or in compilation errors. Instead of using this function use the
// assignment operator.
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
template< typename VT >  // Type of the right-hand side sparse vector
inline void CUDAManagedVector<Type,TF>::multAssign( const SparseVector<VT,TF>& rhs )
{
   BLAZE_INTERNAL_ASSERT( size_ == (~rhs).size(), "Invalid vector sizes" );

   const CUDAManagedVector tmp( serial( *this ) );

   reset();

   for( auto element=(~rhs).begin(); element!=(~rhs).end(); ++element )
      v_[element->index()] = tmp[element->index()] * element->value();
}
//*************************************************************************************************




//=================================================================================================
//
//  CUDAMANAGEDVECTOR OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name CUDAManagedVector operators */
//@{
template< typename Type, bool TF >
void reset( CUDAManagedVector<Type,TF>& v );

template< typename Type, bool TF >
void clear( CUDAManagedVector<Type,TF>& v );

template< bool RF, typename Type, bool TF >
bool isDefault( const CUDAManagedVector<Type,TF>& v );

template< typename Type, bool TF >
bool isIntact( const CUDAManagedVector<Type,TF>& v ) noexcept;

template< typename Type, bool TF >
void swap( CUDAManagedVector<Type,TF>& a, CUDAManagedVector<Type,TF>& b ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Resetting the given dynamic vector.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be resetted.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void reset( CUDAManagedVector<Type,TF>& v )
{
   v.reset();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Clearing the given dynamic vector.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be cleared.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void clear( CUDAManagedVector<Type,TF>& v )
{
   v.clear();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the given dynamic vector is in default state.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be tested for its default state.
// \return \a true in case the given vector's size is zero, \a false otherwise.
//
// This function checks whether the dynamic vector is in default (constructed) state, i.e. if
// it's size is 0. In case it is in default state, the function returns \a true, else it will
// return \a false. The following example demonstrates the use of the \a isDefault() function:

   \code
   CUDAManagedVector<int> a;
   // ... Resizing and initialization
   if( isDefault( a ) ) { ... }
   \endcode

// Optionally, it is possible to switch between strict semantics (strict) and relaxed
// semantics (relaxed):

   \code
   if( isDefault<relaxed>( a ) ) { ... }
   \endcode
*/
template< bool RF        // Relaxation flag
        , typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool isDefault( const CUDAManagedVector<Type,TF>& v )
{
   return ( v.size() == 0UL );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns whether the invariants of the given dynamic vector are intact.
// \ingroup dynamic_vector
//
// \param v The dynamic vector to be tested.
// \return \a true in case the given vector's invariants are intact, \a false otherwise.
//
// This function checks whether the invariants of the dynamic vector are intact, i.e. if its
// state is valid. In case the invariants are intact, the function returns \a true, else it
// will return \a false. The following example demonstrates the use of the \a isIntact()
// function:

   \code
   CUDAManagedVector<int> a;
   // ... Resizing and initialization
   if( isIntact( a ) ) { ... }
   \endcode
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline bool isIntact( const CUDAManagedVector<Type,TF>& v ) noexcept
{
   return v.isIntact();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Swapping the contents of two vectors.
// \ingroup dynamic_vector
//
// \param a The first vector to be swapped.
// \param b The second vector to be swapped.
// \return void
*/
template< typename Type  // Data type of the vector
        , bool TF >      // Transpose flag
inline void swap( CUDAManagedVector<Type,TF>& a, CUDAManagedVector<Type,TF>& b ) noexcept
{
   a.swap( b );
}
//*************************************************************************************************


//=================================================================================================
//
//  HASCONSTDATAACCESS SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF >
struct HasConstDataAccess< CUDAManagedVector<T,TF> >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************



//=================================================================================================
//
//  ISCUDAENABLED SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, bool TF >
struct IsCUDAEnabled< CUDAManagedVector<T,TF> >
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
template< typename T, bool TF >
struct HasMutableDataAccess< CUDAManagedVector<T,TF> >
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
template< typename T, bool TF >
struct IsAligned< CUDAManagedVector<T,TF> >
   : public TrueType
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
template< typename T, bool TF >
struct IsContiguous< CUDAManagedVector<T,TF> >
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
template< typename T, bool TF >
struct IsPadded< CUDAManagedVector<T,TF> >
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
template< typename T, bool TF >
struct IsResizable< CUDAManagedVector<T,TF> >
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
template< typename T, bool TF >
struct IsShrinkable< CUDAManagedVector<T,TF> >
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
template< typename T1, typename T2 >
struct AddTraitEval2< T1, T2
                    , EnableIf_t< IsVector_v<T1> &&
                                  IsVector_v<T2> &&
                                  ( IsDenseVector_v<T1> || IsDenseVector_v<T2> ) &&
                                  IsCUDAEnabled_v<T1> &&
                                  IsCUDAEnabled_v<T2> > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< AddTrait_t<ET1,ET2>, TransposeFlag_v<T1> >;
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
template< typename T1, typename T2 >
struct SubTraitEval2< T1, T2
                    , EnableIf_t< IsVector_v<T1> &&
                                  IsVector_v<T2> &&
                                  ( IsDenseVector_v<T1> || IsDenseVector_v<T2> ) &&
                                  IsCUDAEnabled_v<T1> &&
                                  IsCUDAEnabled_v<T2> > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< SubTrait_t<ET1,ET2>, TransposeFlag_v<T1> >;
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
template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsDenseVector_v<T1> &&
                                   IsNumeric_v<T2> &&
                                   ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                   ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;

   using Type = CUDAManagedVector< MultTrait_t<ET1,T2>, TransposeFlag_v<T1> >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsNumeric_v<T1> &&
                                   IsDenseVector_v<T2> &&
                                   ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                   ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) > >
{
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< MultTrait_t<T1,ET2>, TransposeFlag_v<T2> >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< ( ( IsRowVector_v<T1> && IsRowVector_v<T2> ) ||
                                     ( IsColumnVector_v<T1> && IsColumnVector_v<T2> ) ) &&
                                   IsDenseVector_v<T1> &&
                                   IsDenseVector_v<T2> &&
                                   ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                   ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                   ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
                                   ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< MultTrait_t<ET1,ET2>, TransposeFlag_v<T1> >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsMatrix_v<T1> &&
                                   IsColumnVector_v<T2> &&
                                   ( IsDenseMatrix_v<T1> || IsDenseVector_v<T2> ) &&
                                   ( Size_v<T1,0UL> == DefaultSize_v &&
                                     ( !IsSquare_v<T1> || Size_v<T2,0UL> == DefaultSize_v ) ) &&
                                   ( MaxSize_v<T1,0UL> == DefaultMaxSize_v &&
                                     ( !IsSquare_v<T1> || MaxSize_v<T2,0UL> == DefaultMaxSize_v ) ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< MultTrait_t<ET1,ET2>, false >;
};

template< typename T1, typename T2 >
struct MultTraitEval2< T1, T2
                     , EnableIf_t< IsRowVector_v<T1> &&
                                   IsMatrix_v<T2> &&
                                   ( IsDenseVector_v<T1> || IsDenseMatrix_v<T2> ) &&
                                   ( Size_v<T2,1UL> == DefaultSize_v &&
                                     ( !IsSquare_v<T2> || Size_v<T1,0UL> == DefaultSize_v ) ) &&
                                   ( MaxSize_v<T2,1UL> == DefaultMaxSize_v &&
                                     ( !IsSquare_v<T2> || MaxSize_v<T1,0UL> == DefaultMaxSize_v ) ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< MultTrait_t<ET1,ET2>, true >;
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
template< typename T1, typename T2 >
struct DivTraitEval2< T1, T2
                           , EnableIf_t< IsDenseVector_v<T1> &&
                                                IsNumeric_v<T2> &&
                                                IsCUDAEnabled_v<T1> > >
{
   using ET1 = ElementType_t<T1>;

   using Type = CUDAManagedVector< DivTrait_t<ET1,T2>, TransposeFlag_v<T1> >;
};

template< typename T1, typename T2 >
struct DivTraitEval2< T1, T2
                    , EnableIf_t< IsDenseVector_v<T1> &&
                                  IsDenseVector_v<T2> &&
                                  ( Size_v<T1,0UL> == DefaultSize_v ) &&
                                  ( Size_v<T2,0UL> == DefaultSize_v ) &&
                                  ( MaxSize_v<T1,0UL> == DefaultMaxSize_v ) &&
                                  ( MaxSize_v<T2,0UL> == DefaultMaxSize_v ) > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< DivTrait_t<ET1,ET2>, TransposeFlag_v<T1> >;
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
template< typename T, typename OP >
struct UnaryMapTraitEval2< T, OP
                            , EnableIf_t< IsDenseVector_v<T> &&
                                          IsCUDAEnabled_v<T> > >

                                       //Size_v<T,0UL> == DefaultSize_v &&
                                       //MaxSize_v<T,0UL> == DefaultMaxSize_v > >
{
   using ET = ElementType_t<T>;
   using Type = CUDAManagedVector< MapTrait_t<ET,OP>, TransposeFlag_v<T> >;
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T1, typename T2, typename OP >
struct BinaryMapTraitEval2< T1, T2, OP
                                 , EnableIf_t< IsVector_v<T1> &&
                                                      IsVector_v<T2> &&
                                                      IsCUDAEnabled_v<T1> &&
                                                      IsCUDAEnabled_v<T2> > >

                                        //Size_v<T1,0UL> == DefaultSize_v &&
                                        //Size_v<T2,0UL> == DefaultSize_v &&
                                        //MaxSize_v<T1,0UL> == DefaultMaxSize_v &&
                                        //MaxSize_v<T2,0UL> == DefaultMaxSize_v > >
{
   using ET1 = ElementType_t<T1>;
   using ET2 = ElementType_t<T2>;

   using Type = CUDAManagedVector< MapTrait_t<ET1,ET2,OP>, TransposeFlag_v<T1> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  REDUCETRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T, typename OP, size_t RF >
struct PartialReduceTraitEval2< T, OP, RF
                              , EnableIf_t< IsMatrix_v<T> &&
                                            ( Size_v<T,0UL> == DefaultSize_v ||
                                              Size_v<T,1UL> == DefaultSize_v ) &&
                                            ( MaxSize_v<T,0UL> == DefaultMaxSize_v ||
                                              MaxSize_v<T,1UL> == DefaultMaxSize_v ) > >
{
   static constexpr bool TF = ( RF == 0UL );

   using Type = CUDAManagedVector< ElementType_t<T>, TF >;
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
template< typename T1, bool TF, typename T2 >
struct HighType< CUDAManagedVector<T1,TF>, CUDAManagedVector<T2,TF> >
{
   using Type = CUDAManagedVector< typename HighType<T1,T2>::Type, TF >;
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
template< typename T1, bool TF, typename T2 >
struct LowType< CUDAManagedVector<T1,TF>, CUDAManagedVector<T2,TF> >
{
   using Type = CUDAManagedVector< typename LowType<T1,T2>::Type, TF >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SUBVECTORTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct SubvectorTraitEval2< VT, inf, inf
                                 , EnableIf_t< IsDenseVector_v<VT> &&
                                   Size_v<VT,0UL> == DefaultSize_v &&
                                   MaxSize_v<VT,0UL> == DefaultMaxSize_v > >
{
   using Type = CUDAManagedVector< RemoveConst_t< ElementType_t<VT> >, TransposeFlag_v<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ELEMENTSTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename VT >
struct ElementsTraitEval2< VT, 0UL
                                , EnableIf_t< IsDenseVector_v<VT> &&
                                  Size_v<VT,0UL> == DefaultSize_v &&
                                  MaxSize_v<VT,0UL> == DefaultMaxSize_v > >
{
   using Type = CUDAManagedVector< RemoveConst_t< ElementType_t<VT> >, TransposeFlag_v<VT> >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  ROWTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t I >
struct RowTraitEval2< MT, I
                    , EnableIf_t< IsDenseMatrix_v<MT> &&
                                  Size_v<MT,1UL> == DefaultSize_v &&
                                  MaxSize_v<MT,1UL> == DefaultMaxSize_v > >
{
   using Type = CUDAManagedVector< RemoveConst_t< ElementType_t<MT> >, true >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  COLUMNTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, size_t I >
struct ColumnTraitEval2< MT, I
                       , EnableIf_t< IsDenseMatrix_v<MT> &&
                                     Size_v<MT,0UL> == DefaultSize_v &&
                                     MaxSize_v<MT,0UL> == DefaultMaxSize_v > >
{
   using Type = CUDAManagedVector< RemoveConst_t< ElementType_t<MT> >, false >;
};
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  BANDTRAIT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename MT, ptrdiff_t I >
struct BandTraitEval2< MT, I
                     , EnableIf_t< IsDenseMatrix_v<MT> &&
                                   ( Size_v<MT,0UL> == DefaultSize_v ||
                                     Size_v<MT,1UL> == DefaultSize_v ) &&
                                   ( MaxSize_v<MT,0UL> == DefaultMaxSize_v ||
                                     MaxSize_v<MT,1UL> == DefaultMaxSize_v ) > >
{
   using Type = CUDAManagedVector< RemoveConst_t< ElementType_t<MT> >, defaultTransposeFlag >;
};
/*! \endcond */
//*************************************************************************************************

} // namespace blaze

#endif

