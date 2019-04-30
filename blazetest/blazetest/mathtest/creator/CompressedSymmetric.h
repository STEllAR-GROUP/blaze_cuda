//=================================================================================================
/*!
//  \file blazetest/mathtest/creator/CompressedSymmetric.h
//  \brief Specialization of the Creator class template for SymmetricMatrix<CompressedMatrix>
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

#ifndef _BLAZETEST_MATHTEST_CREATOR_COMPRESSEDSYMMETRIC_H_
#define _BLAZETEST_MATHTEST_CREATOR_COMPRESSEDSYMMETRIC_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <stdexcept>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/SymmetricMatrix.h>
#include <blaze/util/Random.h>
#include <blazetest/mathtest/creator/Default.h>
#include <blazetest/mathtest/creator/Policies.h>
#include <blazetest/system/Types.h>


namespace blazetest {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Specialization of the Creator class template for symmetric compressed matrices.
//
// This specialization of the Creator class template is able to create random symmetric compressed
// matrices.
*/
template< typename T  // Element type of the compressed matrix
        , bool SO >   // Storage order of the compressed matrix
class Creator< blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > >
{
 public:
   //**Type definitions****************************************************************************
   //! Type to be created by the Creator.
   using Type = blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> >;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline Creator( const Creator<T>& elementCreator = Creator<T>() );
   explicit inline Creator( size_t n, size_t nonzeros,
                            const Creator<T>& elementCreator = Creator<T>() );
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Operators***********************************************************************************
   /*!\name Operators */
   //@{
   // No explicitly declared copy assignment operator.

   blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > operator()() const;

   template< typename CP >
   blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > operator()( const CP& policy ) const;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   size_t n_;         //!< The number of rows and columns of the symmetric compressed matrix.
   size_t nonzeros_;  //!< The number of non-zero elements in the symmetric compressed matrix.
   Creator<T> ec_;    //!< Creator for the elements of the symmetric compressed matrix.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the creator specialization for SymmetricMatrix<CompressedMatrix>.
//
// \param elementCreator The creator for the elements of the symmetric compressed matrix.
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename T  // Element type of the compressed matrix
        , bool SO >   // Storage order of the compressed matrix
inline Creator< blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > >::Creator( const Creator<T>& elementCreator )
   : n_( 3UL )              // The number of rows and columns of the symmetric compressed matrix
   , nonzeros_( 3UL )       // The total number of non-zero elements in the symmetric compressed matrix
   , ec_( elementCreator )  // Creator for the elements of the symmetric compressed matrix
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for the creator specialization for SymmetricMatrix<CompressedMatrix>.
//
// \param n The number of rows and columns of the compressed matrix.
// \param nonzeros The number of non-zero elements in the compressed matrix.
// \param elementCreator The creator for the elements of the compressed matrix.
// \exception std::invalid_argument Invalid number of non-zero elements.
*/
template< typename T  // Element type of the compressed matrix
        , bool SO >   // Storage order of the compressed matrix
inline Creator< blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > >::Creator( size_t n, size_t nonzeros,
                                                                                     const Creator<T>& elementCreator )
   : n_( n )                // The number of rows and columns of the symmetric compressed matrix
   , nonzeros_( nonzeros )  // The total number of non-zero elements in the symmetric compressed matrix
   , ec_( elementCreator )  // Creator for the elements of the symmetric compressed matrix
{
   if( n_ * n_ < nonzeros_ )
      throw std::invalid_argument( "Invalid number of non-zero elements" );
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Returns a randomly created symmetric compressed matrix.
//
// \return The randomly generated symmetric compressed matrix.
*/
template< typename T  // Element type of the compressed matrix
        , bool SO >   // Storage order of the compressed matrix
inline blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> >
   Creator< blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > >::operator()() const
{
   return (*this)( Default() );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Returns a randomly created symmetric compressed matrix.
//
// \param policy The creation policy for the elements of fundamental data type.
// \return The randomly generated symmetric compressed matrix.
*/
template< typename T     // Element type of the compressed matrix
        , bool SO >      // Storage order of the compressed matrix
template< typename CP >  // Creation policy
inline blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> >
   Creator< blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > >::operator()( const CP& policy ) const
{
   blaze::SymmetricMatrix< blaze::CompressedMatrix<T,SO> > matrix( n_, nonzeros_ );
   while( matrix.nonZeros() < nonzeros_ )
      matrix( blaze::rand<size_t>(0UL,n_-1UL), blaze::rand<size_t>(0UL,n_-1UL) ) = ec_( policy );
   return matrix;
}
//*************************************************************************************************

} // namespace blazetest

#endif
