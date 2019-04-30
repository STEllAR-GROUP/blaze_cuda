//=================================================================================================
/*!
//  \file src/mathtest/densematrix/StrictlyUpperTest.cpp
//  \brief Source file for the strictly upper DenseMatrix operation test
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


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstdlib>
#include <iostream>
#include <blaze/math/dense/DenseMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/StrictlyUpperMatrix.h>
#include <blazetest/mathtest/densematrix/StrictlyUpperTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace densematrix {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the StrictlyUpperTest class test.
//
// \exception std::runtime_error Operation error detected.
*/
StrictlyUpperTest::StrictlyUpperTest()
{
   testIsSymmetric();
   testIsHermitian();
   testIsUniform();
   testIsZero();
   testIsLower();
   testIsUniLower();
   testIsStrictlyLower();
   testIsUpper();
   testIsUniUpper();
   testIsStrictlyUpper();
   testIsDiagonal();
   testIsIdentity();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the \c isSymmetric() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSymmetric() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsSymmetric()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isSymmetric()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isSymmetric( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isSymmetric( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isSymmetric()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isSymmetric( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isSymmetric( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSymmetric evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isHermitian() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isHermitian() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsHermitian()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isHermitian()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<cplx,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isHermitian( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<cplx,blaze::rowMajor> > mat( 3UL );
         mat(0,2).imag( 4 );
         mat(1,2).imag( 5 );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isHermitian( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isHermitian()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<cplx,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isHermitian( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<cplx,blaze::columnMajor> > mat( 3UL );
         mat(0,2).imag( 4 );
         mat(1,2).imag( 5 );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isHermitian( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isHermitian evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniform() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniform() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsUniform()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isUniform()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniform( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniform( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isUniform()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniform( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isUniform( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniform evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isZero() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isZero() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsZero()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isZero()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isZero( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isZero evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isZero( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isZero evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isZero()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isZero( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isZero evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isZero( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isZero evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isLower() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isLower() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsLower()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isLower()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isLower( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isLower()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isLower( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniLower() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniLower() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsUniLower()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isUniLower()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isUniLower()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isUniLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyLower() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyLower() function for dense matrices. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsStrictlyLower()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isStrictlyLower()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isStrictlyLower( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper triangular matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isStrictlyLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isStrictlyLower()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isStrictlyLower( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper triangular matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isStrictlyLower( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyLower evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUpper() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUpper() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsUpper()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isUpper()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isUpper()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isUniUpper() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isUniUpper() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsUniUpper()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isUniUpper()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniUpper( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniUpper( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isUniUpper()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isUniUpper( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isUniUpper( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isUniUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isStrictlyUpper() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isStrictlyUpper() function for dense matrices. In case an
// error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsStrictlyUpper()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isStrictlyUpper()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isStrictlyUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isStrictlyUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isStrictlyUpper()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isStrictlyUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isStrictlyUpper( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isStrictlyUpper evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDiagonal() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDiagonal() function for dense matrices. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsDiagonal()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isDiagonal()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isDiagonal( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isDiagonal( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isDiagonal()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isDiagonal( mat ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isDiagonal( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDiagonal evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isIdentity() function for dense matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isIdentity() function for dense matrices. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void StrictlyUpperTest::testIsIdentity()
{
   //=====================================================================================
   // Row-majormatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isIdentity()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isIdentity( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::rowMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 1UL );
         checkNonZeros( mat, 1UL, 1UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isIdentity( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-majormatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isIdentity()";

      // Default strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 0UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 0UL );

         if( isIdentity( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Strictly upper matrix
      {
         blaze::StrictlyUpperMatrix< blaze::DynamicMatrix<int,blaze::columnMajor> > mat( 3UL );
         mat(0,2) = 4;
         mat(1,2) = 5;

         checkRows    ( mat, 3UL );
         checkColumns ( mat, 3UL );
         checkCapacity( mat, 9UL );
         checkNonZeros( mat, 2UL );
         checkNonZeros( mat, 0UL, 0UL );
         checkNonZeros( mat, 1UL, 0UL );
         checkNonZeros( mat, 2UL, 2UL );

         if( isIdentity( mat ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isIdentity evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************

} // namespace densematrix

} // namespace mathtest

} // namespace blazetest




//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
int main()
{
   std::cout << "   Running strictly upper DenseMatrix operation test..." << std::endl;

   try
   {
      RUN_DENSEMATRIX_STRICTLY_UPPER_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during strictly upper DenseMatrix operation test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
