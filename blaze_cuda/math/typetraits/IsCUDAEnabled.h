//=================================================================================================
/*!
//  \file blaze/math/typetraits/IsCUDAEnabled.h
//  \brief Header file for the IsCUDAEnabled type trait
//
//  Copyright (C) 2012-2019 Klaus Iglberger - All Rights Reserved
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

#ifndef _BLAZE_TENSOR_MATH_TYPETRAITS_ISCUDAENABLED_H_
#define _BLAZE_TENSOR_MATH_TYPETRAITS_ISCUDAENABLED_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/AlwaysFalse.h>
#include <blaze/util/typetraits/HasMember.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief First auxiliary helper struct for the IsCUDAEnabled type trait.
// \ingroup math_type_traits
*/
BLAZE_CREATE_HAS_DATA_OR_FUNCTION_MEMBER_TYPE_TRAIT( HasCUDAEnabled, cudaEnabled );
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Second auxiliary helper struct for the IsCUDAEnabled type trait.
// \ingroup math_type_traits
*/
template< typename T, typename... Args >
struct GetCUDAEnabled {
   static constexpr bool test( bool (*fnc)() ) { return fnc(); }
   static constexpr bool test( bool b ) { return b; }
   static constexpr bool value = test( T::template cudaEnabled<Args...> );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the GetCUDAEnabled class template for non-templates.
// \ingroup math_type_traits
*/
template< typename T >
struct GetCUDAEnabled<T> {
   static constexpr bool test( bool (*fnc)() ) { return fnc(); }
   static constexpr bool test( bool b ) { return b; }
   static constexpr bool value = test( T::cudaEnabled );
};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Compile time check for data types.
// \ingroup math_type_traits
//
// This type trait tests whether or not the given data type \a T is a CUDA-enabled data type
// (i.e. provides the according CUDA-related member functions, such as \c load(), store(), etc).
// If the type is CUDA-enabled, the \a value member constant is set to \a true, the nested type
// definition \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise \a value
// is set to \a false, \a Type is \a FalseType, and the class derives from \a FalseType.
*/
template< typename T, typename... Args >
struct IsCUDAEnabled
   : public BoolConstant< If_t< HasCUDAEnabled_v<T>
                              , GetCUDAEnabled<T,Args...>
                              , AlwaysFalse<T> >::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the IsCUDAEnabled type trait.
// \ingroup type_traits
//
// The IsCUDAEnabled_v variable template provides a convenient shortcut to access the nested
// \a value of the IsCUDAEnabled class template. For instance, given the type \a T the
// following two statements are identical:

   \code
   constexpr bool value1 = blaze::IsCUDAEnabled<T>::value;
   constexpr bool value2 = blaze::IsCUDAEnabled_v<T>;
   \endcode
*/
template< typename T, typename... Args >
constexpr bool IsCUDAEnabled_v = IsCUDAEnabled<T,Args...>::value;
//*************************************************************************************************

} // namespace blaze

#endif
