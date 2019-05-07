//=================================================================================================
/*!
//  \file blaze_cuda/math/typetraits/IsCUDAPack.h
//  \brief Header file for the IsCUDAPack type trait
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

#ifndef _BLAZE_CUDA_MATH_TYPETRAITS_ISCUDAPACK_H_
#define _BLAZE_CUDA_MATH_TYPETRAITS_ISCUDAPACK_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/cuda/CUDAPack.h>
#include <blaze/util/FalseType.h>
#include <blaze/util/TrueType.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the IsCUDAPack type trait.
// \ingroup math_type_traits
*/
template< typename T >
struct IsCUDAPackHelper
{
 private:
   //**********************************************************************************************
   static T* create();

   template< typename U >
   static TrueType test( const CUDAPack<U>* );

   template< typename U >
   static TrueType test( const volatile CUDAPack<U>* );

   static FalseType test( ... );
   //**********************************************************************************************

 public:
   //**********************************************************************************************
   using Type = decltype( test( create() ) );
   //**********************************************************************************************
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time check for CUDA data types.
// \ingroup math_type_traits
//
// This type trait tests whether the given data type is a Blaze CUDA packed data type. The
// following types are considered valid CUDA packed types:
//
// <ul>
//    <li>Basic CUDA packed data types:</li>
//    <ul>
//       <li>CUDAint8</li>
//       <li>CUDAint16</li>
//       <li>CUDAint32</li>
//       <li>CUDAint64</li>
//       <li>CUDAfloat</li>
//       <li>CUDAdouble</li>
//       <li>CUDAcint8</li>
//       <li>CUDAcint16</li>
//       <li>CUDAcint32</li>
//       <li>CUDAcint64</li>
//       <li>CUDAcfloat</li>
//       <li>CUDAcdouble</li>
//    </ul>
//    <li>Derived CUDA packed data types:</li>
//    <ul>
//       <li>CUDAshort</li>
//       <li>CUDAushort</li>
//       <li>CUDAint</li>
//       <li>CUDAuint</li>
//       <li>CUDAlong</li>
//       <li>CUDAulong</li>
//       <li>CUDAcshort</li>
//       <li>CUDAcushort</li>
//       <li>CUDAcint</li>
//       <li>CUDAcuint</li>
//       <li>CUDAclong</li>
//       <li>CUDAculong</li>
//    </ul>
// </ul>
//
// In case the data type is a CUDA data type, the \a value member constant is set to \a true,
// the nested type definition \a Type is \a TrueType, and the class derives from \a TrueType.
// Otherwise \a value is set to \a false, \a Type is \a FalseType, and the class derives from
// \a FalseType. Examples:

   \code
   blaze::IsCUDAPack< CUDAint32 >::value          // Evaluates to 1
   blaze::IsCUDAPack< const CUDAdouble >::Type    // Results in TrueType
   blaze::IsCUDAPack< volatile CUDAint >          // Is derived from TrueType
   blaze::IsCUDAPack< int >::value                // Evaluates to 0
   blaze::IsCUDAPack< const double >::Type        // Results in FalseType
   blaze::IsCUDAPack< volatile complex<double> >  // Is derived from FalseType
   \endcode
*/
template< typename T >
struct IsCUDAPack
   : public IsCUDAPackHelper<T>::Type
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the IsCUDAPack type trait for references.
// \ingroup math_type_traits
*/
template< typename T >
struct IsCUDAPack<T&>
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsCUDAPack type trait.
// \ingroup type_traits
//
// The IsCUDAPack_v variable template provides a convenient shortcut to access the nested
// \a value of the IsCUDAPack class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::IsCUDAPack<T>::value;
   constexpr bool value2 = blaze::IsCUDAPack_v<T>;
   \endcode
*/
template< typename T >
constexpr bool IsCUDAPack_v = IsCUDAPack<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
