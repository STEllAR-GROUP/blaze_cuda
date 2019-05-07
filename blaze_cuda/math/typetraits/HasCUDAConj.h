//=================================================================================================
/*!
//  \file blaze_cuda/math/typetraits/HasCUDAConj.h
//  \brief Header file for the HasCUDAConj type trait
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

#ifndef _BLAZE_CUDA_MATH_TYPETRAITS_HASCUDACONJ_H_
#define _BLAZE_CUDA_MATH_TYPETRAITS_HASCUDACONJ_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <blaze/math/typetraits/HasCUDAMult.h>
#include <blaze/util/Complex.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/Decay.h>
#include <blaze/util/typetraits/IsFloatingPoint.h>
#include <blaze/util/typetraits/IsNumeric.h>
#include <blaze/util/typetraits/IsSigned.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the HasCUDAConj type trait.
// \ingroup math_type_traits
*/
template< typename T         // Type of the operand
        , typename = void >  // Restricting condition
struct HasCUDAConjHelper
   : public BoolConstant< IsNumeric_v<T> >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
template< typename T >
struct HasCUDAConjHelper< complex<T> >
   : public BoolConstant< IsNumeric_v<T> && IsSigned_v<T> &&
                          ( ( !bool( BLAZE_AVX512F_MODE  ) && HasCUDAMult_v<T,T> && ( IsFloatingPoint_v<T> || sizeof(T) <= 4UL ) ) ||
                            (  bool( BLAZE_AVX512F_MODE  ) && IsFloatingPoint_v<T> ) ||
                            (  bool( BLAZE_AVX512BW_MODE ) && sizeof(T) == 2UL ) ||
                            (  bool( BLAZE_AVX512F_MODE  ) && sizeof(T) >= 4UL ) ) >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Availability of a CUDA conjugate operation for the given data type.
// \ingroup math_type_traits
//
// Depending on the available instruction set (SSE, SSE2, SSE3, SSE4, AVX, AVX2, MIC, ...) and
// the used compiler, this type trait provides the information whether a CUDA conjugate operation
// exists for the given data type \a T (ignoring the cv-qualifiers). In case the CUDA operation
// is available, the \a value member constant is set to \a true, the nested type definition
// \a Type is \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to
// \a false, \a Type is \a FalseType, and the class derives from \a FalseType. The following
// example assumes that AVX is available:

   \code
   blaze::HasCUDAConj< int >::value             // Evaluates to 1
   blaze::HasCUDAConj< double >::Type           // Results in TrueType
   blaze::HasCUDAConj< complex<float> >         // Is derived from TrueType
   blaze::HasCUDAConj< complex<bool> >::value   // Evaluates to 0
   blaze::HasCUDAConj< complex<int> >::Type     // Results in FalseType
   blaze::HasCUDAConj< complex<unsigned int> >  // Is derived from FalseType
   \endcode
*/
template< typename T >  // Type of the operand
struct HasCUDAConj
   : public BoolConstant< HasCUDAConjHelper< Decay_t<T> >::value >
{};
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the HasCUDAConj type trait.
// \ingroup type_traits
//
// The HasCUDAConj_v variable template provides a convenient shortcut to access the nested
// \a value of the HasCUDAConj class template. For instance, given the type \a T the following
// two statements are identical:

   \code
   constexpr bool value1 = blaze::HasCUDAConj<T>::value;
   constexpr bool value2 = blaze::HasCUDAConj_v<T>;
   \endcode
*/
template< typename T >  // Type of the operand
constexpr bool HasCUDAConj_v = HasCUDAConj<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
