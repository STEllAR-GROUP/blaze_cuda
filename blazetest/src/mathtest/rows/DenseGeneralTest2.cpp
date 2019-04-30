//=================================================================================================
/*!
//  \file src/mathtest/rows/DenseGeneralTest2.cpp
//  \brief Source file for the Rows dense general test (part 2)
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
#include <memory>
#include <blaze/math/Views.h>
#include <blazetest/mathtest/rows/DenseGeneralTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace rows {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the Rows dense general test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseGeneralTest::DenseGeneralTest()
   : mat_ ( 5UL, 4UL )
   , tmat_( 5UL, 4UL )
{
   testScaling();
   testFunctionCall();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
   testTranspose();
   testCTranspose();
   testIsDefault();
   testIsSame();
   testSubmatrix();
   testRow();
   testRows();
   testColumn();
   testColumns();
   testBand();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of all Rows (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the Rows
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testScaling()
{
   //=====================================================================================
   // Row-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s)";

      initialize();

      auto rs = blaze::rows( mat_, { 2UL, 3UL } );

      rs *= 3;

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != -6 || rs(0,1) !=  0 || rs(0,2) != -9 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 12 || rs(1,2) != 15 || rs(1,3) != -18 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -6   0  -9   0 )\n(  0  12  15 -18 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=   0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=   0 ||
          mat_(2,0) != -6 || mat_(2,1) !=  0 || mat_(2,2) != -9 || mat_(2,3) !=   0 ||
          mat_(3,0) !=  0 || mat_(3,1) != 12 || mat_(3,2) != 15 || mat_(3,3) != -18 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -6   0  -9   0 )\n"
                                     "(  0  12  15 -18 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M*s)";

      initialize();

      auto rs = blaze::rows( mat_, { 2UL, 3UL } );

      rs = rs * 3;

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != -6 || rs(0,1) !=  0 || rs(0,2) != -9 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 12 || rs(1,2) != 15 || rs(1,3) != -18 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -6   0  -9   0 )\n(  0  12  15 -18 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=   0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=   0 ||
          mat_(2,0) != -6 || mat_(2,1) !=  0 || mat_(2,2) != -9 || mat_(2,3) !=   0 ||
          mat_(3,0) !=  0 || mat_(3,1) != 12 || mat_(3,2) != 15 || mat_(3,3) != -18 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -6   0  -9   0 )\n"
                                     "(  0  12  15 -18 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=s*M)";

      initialize();

      auto rs = blaze::rows( mat_, { 2UL, 3UL } );

      rs = 3 * rs;

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != -6 || rs(0,1) !=  0 || rs(0,2) != -9 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 12 || rs(1,2) != 15 || rs(1,3) != -18 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -6   0  -9   0 )\n(  0  12  15 -18 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=   0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=   0 ||
          mat_(2,0) != -6 || mat_(2,1) !=  0 || mat_(2,2) != -9 || mat_(2,3) !=   0 ||
          mat_(3,0) !=  0 || mat_(3,1) != 12 || mat_(3,2) != 15 || mat_(3,3) != -18 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -6   0  -9   0 )\n"
                                     "(  0  12  15 -18 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M/=s)";

      initialize();

      auto rs = blaze::rows( mat_, { 2UL, 3UL } );

      rs /= 0.5;

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != -4 || rs(0,1) != 0 || rs(0,2) != -6 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 8 || rs(1,2) != 10 || rs(1,3) != -12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -4   0  -6   0 )\n(  0   8  10 -12 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=   0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=   0 ||
          mat_(2,0) != -4 || mat_(2,1) !=  0 || mat_(2,2) != -6 || mat_(2,3) !=   0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) != 10 || mat_(3,3) != -12 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -4   0  -6   0 )\n"
                                     "(  0   8  10 -12 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M/s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M/s)";

      initialize();

      auto rs = blaze::rows( mat_, { 2UL, 3UL } );

      rs = rs / 0.5;

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != -4 || rs(0,1) != 0 || rs(0,2) != -6 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 8 || rs(1,2) != 10 || rs(1,3) != -12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -4   0  -6   0 )\n(  0   8  10 -12 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=   0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=   0 ||
          mat_(2,0) != -4 || mat_(2,1) !=  0 || mat_(2,2) != -6 || mat_(2,3) !=   0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) != 10 || mat_(3,3) != -12 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -4   0  -6   0 )\n"
                                     "(  0   8  10 -12 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major Rows::scale()
   //=====================================================================================

   {
      test_ = "Row-major Rows::scale()";

      initialize();

      // Initialization check
      auto rs = blaze::rows( mat_, { 2UL, 3UL } );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 5UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 3UL );

      if( rs(0,0) != -2 || rs(0,1) != 0 || rs(0,2) != -3 || rs(0,3) !=  0 ||
          rs(1,0) !=  0 || rs(1,1) != 4 || rs(1,2) !=  5 || rs(1,3) != -6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Integral scaling of the matrix
      rs.scale( 2 );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 5UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 3UL );

      if( rs(0,0) != -4 || rs(0,1) != 0 || rs(0,2) != -6 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 8 || rs(1,2) != 10 || rs(1,3) != -12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -4   0  -6   0 )\n(  0   5  10  -12 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the matrix
      rs.scale( 0.5 );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 5UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 3UL );

      if( rs(0,0) != -2 || rs(0,1) != 0 || rs(0,2) != -3 || rs(0,3) !=  0 ||
          rs(1,0) !=  0 || rs(1,1) != 4 || rs(1,2) !=  5 || rs(1,3) != -6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M*=s)";

      initialize();

      auto rs = blaze::rows( tmat_, { 2UL, 3UL } );

      rs *= 3;

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  5UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != -6 || rs(0,1) !=  0 || rs(0,2) != -9 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 12 || rs(1,2) != 15 || rs(1,3) != -18 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -6   0  -9   0 )\n(  0  12  15 -18 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=   0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=   0 ||
          tmat_(2,0) != -6 || tmat_(2,1) !=  0 || tmat_(2,2) != -9 || tmat_(2,3) !=   0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != 12 || tmat_(3,2) != 15 || tmat_(3,3) != -18 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -6   0  -9   0 )\n"
                                     "(  0  12  15 -18 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M=M*s)";

      initialize();

      auto rs = blaze::rows( tmat_, { 2UL, 3UL } );

      rs = rs * 3;

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  5UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != -6 || rs(0,1) !=  0 || rs(0,2) != -9 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 12 || rs(1,2) != 15 || rs(1,3) != -18 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -6   0  -9   0 )\n(  0  12  15 -18 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=   0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=   0 ||
          tmat_(2,0) != -6 || tmat_(2,1) !=  0 || tmat_(2,2) != -9 || tmat_(2,3) !=   0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != 12 || tmat_(3,2) != 15 || tmat_(3,3) != -18 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -6   0  -9   0 )\n"
                                     "(  0  12  15 -18 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M=s*M)";

      initialize();

      auto rs = blaze::rows( tmat_, { 2UL, 3UL } );

      rs = 3 * rs;

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  5UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != -6 || rs(0,1) !=  0 || rs(0,2) != -9 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 12 || rs(1,2) != 15 || rs(1,3) != -18 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -6   0  -9   0 )\n(  0  12  15 -18 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=   0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=   0 ||
          tmat_(2,0) != -6 || tmat_(2,1) !=  0 || tmat_(2,2) != -9 || tmat_(2,3) !=   0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != 12 || tmat_(3,2) != 15 || tmat_(3,3) != -18 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -6   0  -9   0 )\n"
                                     "(  0  12  15 -18 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M/=s)";

      initialize();

      auto rs = blaze::rows( tmat_, { 2UL, 3UL } );

      rs /= 0.5;

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  5UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != -4 || rs(0,1) != 0 || rs(0,2) != -6 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 8 || rs(1,2) != 10 || rs(1,3) != -12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -4   0  -6   0 )\n(  0   8  10 -12 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=   0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=   0 ||
          tmat_(2,0) != -4 || tmat_(2,1) !=  0 || tmat_(2,2) != -6 || tmat_(2,3) !=   0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) !=  8 || tmat_(3,2) != 10 || tmat_(3,3) != -12 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -4   0  -6   0 )\n"
                                     "(  0   8  10 -12 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=M/s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M=M/s)";

      initialize();

      auto rs = blaze::rows( tmat_, { 2UL, 3UL } );

      rs = rs / 0.5;

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  5UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != -4 || rs(0,1) != 0 || rs(0,2) != -6 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 8 || rs(1,2) != 10 || rs(1,3) != -12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -4   0  -6   0 )\n(  0   8  10 -12 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=   0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=   0 ||
          tmat_(2,0) != -4 || tmat_(2,1) !=  0 || tmat_(2,2) != -6 || tmat_(2,3) !=   0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) !=  8 || tmat_(3,2) != 10 || tmat_(3,3) != -12 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0   0   0   0 )\n"
                                     "(  0   1   0   0 )\n"
                                     "( -4   0  -6   0 )\n"
                                     "(  0   8  10 -12 )\n"
                                     "(  7  -8   9  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major Rows::scale()
   //=====================================================================================

   {
      test_ = "Column-major Rows::scale()";

      initialize();

      // Initialization check
      auto rs = blaze::rows( tmat_, { 2UL, 3UL } );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 5UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 3UL );

      if( rs(0,0) != -2 || rs(0,1) != 0 || rs(0,2) != -3 || rs(0,3) !=  0 ||
          rs(1,0) !=  0 || rs(1,1) != 4 || rs(1,2) !=  5 || rs(1,3) != -6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Integral scaling of the matrix
      rs.scale( 2 );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 5UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 3UL );

      if( rs(0,0) != -4 || rs(0,1) != 0 || rs(0,2) != -6 || rs(0,3) !=   0 ||
          rs(1,0) !=  0 || rs(1,1) != 8 || rs(1,2) != 10 || rs(1,3) != -12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -4   0  -6   0 )\n(  0   5  10  -12 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the matrix
      rs.scale( 0.5 );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 5UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 3UL );

      if( rs(0,0) != -2 || rs(0,1) != 0 || rs(0,2) != -3 || rs(0,3) !=  0 ||
          rs(1,0) !=  0 || rs(1,1) != 4 || rs(1,2) !=  5 || rs(1,3) != -6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Rows function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the Rows specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void DenseGeneralTest::testFunctionCall()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::operator()";

      initialize();

      auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      // Assignment to the element (1,1)
      {
         rs(1,1) = 9;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  7UL );
         checkNonZeros( rs  ,  0UL, 1UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 3UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != -3 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 4 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  1  0  0 )\n( -2  9 -3  0 )\n(  0  4  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  9 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (2,1)
      {
         rs(2,1) = 0;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  6UL );
         checkNonZeros( rs  ,  0UL, 1UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 10UL );

         if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != -3 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  1  0  0 )\n( -2  9 -3  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  9 -3  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,2)
      {
         rs(1,2) = 11;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  6UL );
         checkNonZeros( rs  ,  0UL, 1UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 10UL );

         if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  1  0  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != 11 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (0,1)
      {
         rs(0,1) += 3;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  6UL );
         checkNonZeros( rs  ,  0UL, 1UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 10UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4  0  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  4 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != 11 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4  0  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (0,2)
      {
         rs(0,2) -= 6;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  7UL );
         checkNonZeros( rs  ,  0UL, 2UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) != -6 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4 -6  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  4 || mat_(1,2) != -6 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != 11 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4 -6  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,2)
      {
         rs(1,2) *= 2;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  7UL );
         checkNonZeros( rs  ,  0UL, 2UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) != -6 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 22 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4 -6  0 )\n( -2  9 22  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  4 || mat_(1,2) != -6 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != 22 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4 -6  0 )\n"
                                        "( -2  9 22  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (1,2)
      {
         rs(1,2) /= 2;

         checkRows    ( rs  ,  3UL );
         checkColumns ( rs  ,  4UL );
         checkNonZeros( rs  ,  7UL );
         checkNonZeros( rs  ,  0UL, 2UL );
         checkNonZeros( rs  ,  1UL, 3UL );
         checkNonZeros( rs  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) != -6 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4 -6  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  4 || mat_(1,2) != -6 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != 11 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4 -6  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::operator()";

      initialize();

      auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      // Assignment to the element (1,1)
      {
         rs(1,1) = 9;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  7UL );
         checkNonZeros( rs   ,  0UL, 1UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 3UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != -3 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 4 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  1  0  0 )\n( -2  9 -3  0 )\n(  0  4  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  9 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (2,1)
      {
         rs(2,1) = 0;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  6UL );
         checkNonZeros( rs   ,  0UL, 1UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 2UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 10UL );

         if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != -3 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  1  0  0 )\n( -2  9 -3  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  9 -3  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,2)
      {
         rs(1,2) = 11;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  6UL );
         checkNonZeros( rs   ,  0UL, 1UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 2UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 10UL );

         if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  1  0  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != 11 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (0,1)
      {
         rs(0,1) += 3;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  6UL );
         checkNonZeros( rs   ,  0UL, 1UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 2UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 10UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) !=  0 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4  0  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  4 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != 11 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4  0  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (0,2)
      {
         rs(0,2) -= 6;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  7UL );
         checkNonZeros( rs   ,  0UL, 2UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 2UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) != -6 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4 -6  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  4 || tmat_(1,2) != -6 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != 11 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4 -6  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,2)
      {
         rs(1,2) *= 2;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  7UL );
         checkNonZeros( rs   ,  0UL, 2UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 2UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) != -6 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 22 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4 -6  0 )\n( -2  9 22  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  4 || tmat_(1,2) != -6 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != 22 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4 -6  0 )\n"
                                        "( -2  9 22  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (1,2)
      {
         rs(1,2) /= 2;

         checkRows    ( rs   ,  3UL );
         checkColumns ( rs   ,  4UL );
         checkNonZeros( rs   ,  7UL );
         checkNonZeros( rs   ,  0UL, 2UL );
         checkNonZeros( rs   ,  1UL, 3UL );
         checkNonZeros( rs   ,  2UL, 2UL );
         checkRows    ( tmat_,  5UL );
         checkColumns ( tmat_,  4UL );
         checkNonZeros( tmat_, 11UL );

         if( rs(0,0) !=  0 || rs(0,1) != 4 || rs(0,2) != -6 || rs(0,3) !=  0 ||
             rs(1,0) != -2 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) !=  0 ||
             rs(2,0) !=  0 || rs(2,1) != 0 || rs(2,2) !=  5 || rs(2,3) != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n(  0  4 -6  0 )\n( -2  9 11  0 )\n(  0  0  5 -6 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  4 || tmat_(1,2) != -6 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  9 || tmat_(2,2) != 11 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  4 -6  0 )\n"
                                        "( -2  9 11  0 )\n"
                                        "(  0  0  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Rows iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIterator()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Row-major Iterator default constructor";

         RT::Iterator it{};

         if( it != RT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         RT::ConstIterator it{};

         if( it != RT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         auto rs = blaze::rows( mat_, { 2UL } );
         auto it( begin( rs, 0UL ) );

         if( it == end( rs, 0UL ) || *it != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         auto rs = blaze::rows( mat_, { 1UL } );
         const ptrdiff_t number( end( rs, 0UL ) - begin( rs, 0UL ) );

         if( number != 4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         auto rs = blaze::rows( mat_, { 1UL } );
         const ptrdiff_t number( begin( rs, 0UL ) - end( rs, 0UL ) );

         if( number != -4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 2nd row via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         auto rs = blaze::rows( mat_, { 2UL } );
         const ptrdiff_t number( cend( rs, 0UL ) - cbegin( rs, 0UL ) );

         if( number != 4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 2nd row via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         auto rs = blaze::rows( mat_, { 2UL } );
         const ptrdiff_t number( cbegin( rs, 0UL ) - cend( rs, 0UL ) );

         if( number != -4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Row-major read-only access via ConstIterator";

         auto rs = blaze::rows( mat_, { 3UL } );
         auto it ( cbegin( rs, 0UL ) );
         auto end( cend( rs, 0UL ) );

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 3UL;

         if( it == end || *it != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 3UL;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 4UL + it;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Scalar/iterator addition failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Row-major assignment via Iterator";

         auto rs = blaze::rows( mat_, { 0UL } );
         int value = 6;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it = value++;
         }

         if( rs(0,0) != 6 || rs(0,1) != 7 || rs(0,2) != 8 || rs(0,3) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  6 || mat_(0,1) !=  7 || mat_(0,2) !=  8 || mat_(0,3) !=  9 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  6  7  8  9 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         auto rs = blaze::rows( mat_, { 0UL } );
         int value = 2;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it += value++;
         }

         if( rs(0,0) != 8 || rs(0,1) != 10 || rs(0,2) != 12 || rs(0,3) != 14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 8 10 12 14 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  8 || mat_(0,1) != 10 || mat_(0,2) != 12 || mat_(0,3) != 14 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  8 10 12 14 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         auto rs = blaze::rows( mat_, { 0UL } );
         int value = 2;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( rs(0,0) != 6 || rs(0,1) != 7 || rs(0,2) != 8 || rs(0,3) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  6 || mat_(0,1) !=  7 || mat_(0,2) !=  8 || mat_(0,3) !=  9 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  6  7  8  9 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         auto rs = blaze::rows( mat_, { 0UL } );
         int value = 1;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( rs(0,0) != 6 || rs(0,1) != 14 || rs(0,2) != 24 || rs(0,3) != 36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 6 14 24 36 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  6 || mat_(0,1) != 14 || mat_(0,2) != 24 || mat_(0,3) != 36 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  6 14 24 36 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         auto rs = blaze::rows( mat_, { 0UL } );

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( rs(0,0) != 3 || rs(0,1) != 7 || rs(0,2) != 12 || rs(0,3) != 18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 3 7 12 18 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  3 || mat_(0,1) !=  7 || mat_(0,2) != 12 || mat_(0,3) != 18 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  3  7 12 18 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      initialize();

      // Testing the Iterator default constructor
      {
         test_ = "Column-major Iterator default constructor";

         ORT::Iterator it{};

         if( it != ORT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Column-major ConstIterator default constructor";

         ORT::ConstIterator it{};

         if( it != ORT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Column-major Iterator/ConstIterator conversion";

         auto rs = blaze::rows( tmat_, { 2UL } );
         auto it( begin( rs, 0UL ) );

         if( it == end( rs, 0UL ) || *it != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via Iterator (end-begin)
      {
         test_ = "Column-major Iterator subtraction (end-begin)";

         auto rs = blaze::rows( tmat_, { 1UL } );
         const ptrdiff_t number( end( rs, 0UL ) - begin( rs, 0UL ) );

         if( number != 4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via Iterator (begin-end)
      {
         test_ = "Column-major Iterator subtraction (begin-end)";

         auto rs = blaze::rows( tmat_, { 1UL } );
         const ptrdiff_t number( begin( rs, 0UL ) - end( rs, 0UL ) );

         if( number != -4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 2nd row via ConstIterator (end-begin)
      {
         test_ = "Column-major ConstIterator subtraction (end-begin)";

         auto rs = blaze::rows( tmat_, { 2UL } );
         const ptrdiff_t number( cend( rs, 0UL ) - cbegin( rs, 0UL ) );

         if( number != 4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 2nd row via ConstIterator (begin-end)
      {
         test_ = "Column-major ConstIterator subtraction (begin-end)";

         auto rs = blaze::rows( tmat_, { 2UL } );
         const ptrdiff_t number( cbegin( rs, 0UL ) - cend( rs, 0UL ) );

         if( number != -4L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: -4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Column-major read-only access via ConstIterator";

         auto rs = blaze::rows( tmat_, { 3UL } );
         auto it ( cbegin( rs, 0UL ) );
         auto end( cend( rs, 0UL ) );

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         --it;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it == end || *it != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it--;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-decrement failed\n";
            throw std::runtime_error( oss.str() );
         }

         it += 2UL;

         if( it == end || *it != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator addition assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it -= 2UL;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator subtraction assignment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it + 3UL;

         if( it == end || *it != -6 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar addition failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = it - 3UL;

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator/scalar subtraction failed\n";
            throw std::runtime_error( oss.str() );
         }

         it = 4UL + it;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Scalar/iterator addition failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Column-major assignment via Iterator";

         auto rs = blaze::rows( tmat_, { 0UL } );
         int value = 6;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it = value++;
         }

         if( rs(0,0) != 6 || rs(0,1) != 7 || rs(0,2) != 8 || rs(0,3) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  6 || tmat_(0,1) !=  7 || tmat_(0,2) !=  8 || tmat_(0,3) !=  9 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  6  7  8  9 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Column-major addition assignment via Iterator";

         auto rs = blaze::rows( tmat_, { 0UL } );
         int value = 2;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it += value++;
         }

         if( rs(0,0) != 8 || rs(0,1) != 10 || rs(0,2) != 12 || rs(0,3) != 14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 8 10 12 14 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  8 || tmat_(0,1) != 10 || tmat_(0,2) != 12 || tmat_(0,3) != 14 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  8 10 12 14 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Column-major subtraction assignment via Iterator";

         auto rs = blaze::rows( tmat_, { 0UL } );
         int value = 2;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( rs(0,0) != 6 || rs(0,1) != 7 || rs(0,2) != 8 || rs(0,3) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  6 || tmat_(0,1) !=  7 || tmat_(0,2) !=  8 || tmat_(0,3) !=  9 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  6  7  8  9 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Column-major multiplication assignment via Iterator";

         auto rs = blaze::rows( tmat_, { 0UL } );
         int value = 1;

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( rs(0,0) != 6 || rs(0,1) != 14 || rs(0,2) != 24 || rs(0,3) != 36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 6 14 24 36 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  6 || tmat_(0,1) != 14 || tmat_(0,2) != 24 || tmat_(0,3) != 36 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  6 14 24 36 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Column-major division assignment via Iterator";

         auto rs = blaze::rows( tmat_, { 0UL } );

         for( auto it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( rs(0,0) != 3 || rs(0,1) != 7 || rs(0,2) != 12 || rs(0,3) != 18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 3 7 12 18 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  3 || tmat_(0,1) !=  7 || tmat_(0,2) != 12 || tmat_(0,3) != 18 ||
             tmat_(1,0) !=  0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
             tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
             tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
             tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  3  7 12 18 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  4  5 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testNonZeros()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::nonZeros()";

      initialize();

      // Initialization check
      auto rs = blaze::rows( mat_, { 1UL, 2UL } );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 1UL );
      checkNonZeros( rs, 1UL, 2UL );

      if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) != 0 ||
          rs(1,0) != -2 || rs(1,1) != 0 || rs(1,2) != -3 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0  0 )\n( -2  0 -3  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the row selection
      rs(1,2) = 0;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 2UL );
      checkNonZeros( rs, 0UL, 1UL );
      checkNonZeros( rs, 1UL, 1UL );

      if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != 0 ||
          rs(1,0) != -2 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0  0 )\n( -2  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      mat_(2,3) = 5;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 1UL );
      checkNonZeros( rs, 1UL, 2UL );

      if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != 0 ||
          rs(1,0) != -2 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0  0 )\n( -2  0  0  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::nonZeros()";

      initialize();

      // Initialization check
      auto rs = blaze::rows( tmat_, { 1UL, 2UL } );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 1UL );
      checkNonZeros( rs, 1UL, 2UL );

      if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) !=  0 || rs(0,3) != 0 ||
          rs(1,0) != -2 || rs(1,1) != 0 || rs(1,2) != -3 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0  0 )\n( -2  0 -3  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the row selection
      rs(1,2) = 0;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 2UL );
      checkNonZeros( rs, 0UL, 1UL );
      checkNonZeros( rs, 1UL, 1UL );

      if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != 0 ||
          rs(1,0) != -2 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0  0 )\n( -2  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      tmat_(2,3) = 5;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 1UL );
      checkNonZeros( rs, 1UL, 2UL );

      if( rs(0,0) !=  0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != 0 ||
          rs(1,0) != -2 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0  0 )\n( -2  0  0  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testReset()
{
   //=====================================================================================
   // Row-major single element reset
   //=====================================================================================

   {
      test_ = "Row-major reset() function";

      using blaze::reset;
      using blaze::isDefault;

      initialize();

      auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      reset( rs(0,1) );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 5UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  0 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "( -2  0 -3  0 )\n"
                                     "(  0  4  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major reset
   //=====================================================================================

   {
      test_ = "Row-major Rows::reset() (lvalue)";

      initialize();

      auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      reset( rs );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 0UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( !isDefault( rs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 0 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Rows::reset() (rvalue)";

      initialize();

      reset( blaze::rows( mat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 0 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major single element reset
   //=====================================================================================

   {
      test_ = "Column-major reset() function";

      using blaze::reset;
      using blaze::isDefault;

      initialize();

      auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      reset( rs(0,1) );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 5UL );
      checkRows    ( tmat_, 5UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 9UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "( -2  0 -3  0 )\n"
                                     "(  0  4  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major reset
   //=====================================================================================

   {
      test_ = "Column-major Rows::reset() (lvalue)";

      initialize();

      auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      reset( rs );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 0UL );
      checkRows    ( tmat_, 5UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( !isDefault( rs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 0 || tmat_(2,3) !=  0 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ||
          tmat_(4,0) != 7 || tmat_(4,1) != -8 || tmat_(4,2) != 9 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major Rows::reset() (rvalue)";

      initialize();

      reset( blaze::rows( tmat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( tmat_, 5UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 0 || tmat_(2,3) !=  0 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ||
          tmat_(4,0) != 7 || tmat_(4,1) != -8 || tmat_(4,2) != 9 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testClear()
{
   //=====================================================================================
   // Row-major single element clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function";

      using blaze::clear;
      using blaze::isDefault;

      initialize();

      auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      clear( rs(0,1) );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 5UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  0 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "( -2  0 -3  0 )\n"
                                     "(  0  4  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major clear
   //=====================================================================================

   {
      test_ = "Row-major Rows::clear() (lvalue)";

      initialize();

      auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      clear( rs );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 0UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( !isDefault( rs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 0 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Rows::clear() (rvalue)";

      initialize();

      clear( blaze::rows( mat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 0 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major single element clear
   //=====================================================================================

   {
      test_ = "Column-major clear() function";

      using blaze::clear;
      using blaze::isDefault;

      initialize();

      auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      clear( rs(0,1) );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 5UL );
      checkRows    ( tmat_, 5UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 9UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n( -2  0 -3  0 )\n(  0  4  5 -6 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) !=  0 || tmat_(1,1) !=  0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != -2 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  0 ||
          tmat_(3,0) !=  0 || tmat_(3,1) !=  4 || tmat_(3,2) !=  5 || tmat_(3,3) != -6 ||
          tmat_(4,0) !=  7 || tmat_(4,1) != -8 || tmat_(4,2) !=  9 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "( -2  0 -3  0 )\n"
                                     "(  0  4  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major clear
   //=====================================================================================

   {
      test_ = "Column-major Rows::clear() (lvalue)";

      initialize();

      auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      clear( rs );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 0UL );
      checkRows    ( tmat_, 5UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( !isDefault( rs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 0 || tmat_(2,3) !=  0 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ||
          tmat_(4,0) != 7 || tmat_(4,1) != -8 || tmat_(4,2) != 9 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major Rows::clear() (rvalue)";

      initialize();

      clear( blaze::rows( tmat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( tmat_, 5UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 0 || tmat_(2,3) !=  0 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ||
          tmat_(4,0) != 7 || tmat_(4,1) != -8 || tmat_(4,2) != 9 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c transpose() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c transpose() member function of the Rows class
// template. Additionally, it performs a test of self-transpose via the \c trans() function.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testTranspose()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via transpose()";

      initialize();

      auto rs = blaze::rows( mat_, { 3UL, 2UL, 1UL, 4UL } );

      transpose( rs );

      checkRows    ( rs  ,  4UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  , 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  5 || mat_(1,1) != -3 || mat_(1,2) != 0 || mat_(1,3) !=  9 ||
          mat_(2,0) !=  4 || mat_(2,1) !=  0 || mat_(2,2) != 1 || mat_(2,3) != -8 ||
          mat_(3,0) !=  0 || mat_(3,1) != -2 || mat_(3,2) != 0 || mat_(3,3) !=  7 ||
          mat_(4,0) != -6 || mat_(4,1) !=  0 || mat_(4,2) != 0 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via trans()";

      initialize();

      auto rs = blaze::rows( mat_, { 3UL, 2UL, 1UL, 4UL } );

      rs = trans( rs );

      checkRows    ( rs  ,  4UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  , 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  5 || mat_(1,1) != -3 || mat_(1,2) != 0 || mat_(1,3) !=  9 ||
          mat_(2,0) !=  4 || mat_(2,1) !=  0 || mat_(2,2) != 1 || mat_(2,3) != -8 ||
          mat_(3,0) !=  0 || mat_(3,1) != -2 || mat_(3,2) != 0 || mat_(3,3) !=  7 ||
          mat_(4,0) != -6 || mat_(4,1) !=  0 || mat_(4,2) != 0 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major self-transpose via transpose()";

      initialize();

      auto rs = blaze::rows( tmat_, { 3UL, 2UL, 1UL, 4UL } );

      transpose( rs );

      checkRows    ( rs   ,  4UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   , 10UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) !=  5 || tmat_(1,1) != -3 || tmat_(1,2) != 0 || tmat_(1,3) !=  9 ||
          tmat_(2,0) !=  4 || tmat_(2,1) !=  0 || tmat_(2,2) != 1 || tmat_(2,3) != -8 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != -2 || tmat_(3,2) != 0 || tmat_(3,3) !=  7 ||
          tmat_(4,0) != -6 || tmat_(4,1) !=  0 || tmat_(4,2) != 0 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-transpose via trans()";

      initialize();

      auto rs = blaze::rows( tmat_, { 3UL, 2UL, 1UL, 4UL } );

      rs = trans( rs );

      checkRows    ( rs   ,  4UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   , 10UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) !=  5 || tmat_(1,1) != -3 || tmat_(1,2) != 0 || tmat_(1,3) !=  9 ||
          tmat_(2,0) !=  4 || tmat_(2,1) !=  0 || tmat_(2,2) != 1 || tmat_(2,3) != -8 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != -2 || tmat_(3,2) != 0 || tmat_(3,3) !=  7 ||
          tmat_(4,0) != -6 || tmat_(4,1) !=  0 || tmat_(4,2) != 0 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c ctranspose() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c ctranspose() member function of the Rows
// specialization. Additionally, it performs a test of self-transpose via the \c ctrans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testCTranspose()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via ctranspose()";

      initialize();

      auto rs = blaze::rows( mat_, { 3UL, 2UL, 1UL, 4UL } );

      ctranspose( rs );

      checkRows    ( rs  ,  4UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  , 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  5 || mat_(1,1) != -3 || mat_(1,2) != 0 || mat_(1,3) !=  9 ||
          mat_(2,0) !=  4 || mat_(2,1) !=  0 || mat_(2,2) != 1 || mat_(2,3) != -8 ||
          mat_(3,0) !=  0 || mat_(3,1) != -2 || mat_(3,2) != 0 || mat_(3,3) !=  7 ||
          mat_(4,0) != -6 || mat_(4,1) !=  0 || mat_(4,2) != 0 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via ctrans()";

      initialize();

      auto rs = blaze::rows( mat_, { 3UL, 2UL, 1UL, 4UL } );

      rs = ctrans( rs );

      checkRows    ( rs  ,  4UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  , 10UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  5 || mat_(1,1) != -3 || mat_(1,2) != 0 || mat_(1,3) !=  9 ||
          mat_(2,0) !=  4 || mat_(2,1) !=  0 || mat_(2,2) != 1 || mat_(2,3) != -8 ||
          mat_(3,0) !=  0 || mat_(3,1) != -2 || mat_(3,2) != 0 || mat_(3,3) !=  7 ||
          mat_(4,0) != -6 || mat_(4,1) !=  0 || mat_(4,2) != 0 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major self-transpose via ctranspose()";

      initialize();

      auto rs = blaze::rows( tmat_, { 3UL, 2UL, 1UL, 4UL } );

      ctranspose( rs );

      checkRows    ( rs   ,  4UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   , 10UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) !=  5 || tmat_(1,1) != -3 || tmat_(1,2) != 0 || tmat_(1,3) !=  9 ||
          tmat_(2,0) !=  4 || tmat_(2,1) !=  0 || tmat_(2,2) != 1 || tmat_(2,3) != -8 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != -2 || tmat_(3,2) != 0 || tmat_(3,3) !=  7 ||
          tmat_(4,0) != -6 || tmat_(4,1) !=  0 || tmat_(4,2) != 0 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-transpose via ctrans()";

      initialize();

      auto rs = blaze::rows( tmat_, { 3UL, 2UL, 1UL, 4UL } );

      rs = ctrans( rs );

      checkRows    ( rs   ,  4UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   , 10UL );
      checkRows    ( tmat_,  5UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) !=  0 || rs(0,1) != -2 || rs(0,2) != 0 || rs(0,3) !=  7 ||
          rs(1,0) !=  4 || rs(1,1) !=  0 || rs(1,2) != 1 || rs(1,3) != -8 ||
          rs(2,0) !=  5 || rs(2,1) != -3 || rs(2,2) != 0 || rs(2,3) !=  9 ||
          rs(3,0) != -6 || rs(3,1) !=  0 || rs(3,2) != 0 || rs(3,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n(  0 -2  0  7 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) !=  5 || tmat_(1,1) != -3 || tmat_(1,2) != 0 || tmat_(1,3) !=  9 ||
          tmat_(2,0) !=  4 || tmat_(2,1) !=  0 || tmat_(2,2) != 1 || tmat_(2,3) != -8 ||
          tmat_(3,0) !=  0 || tmat_(3,1) != -2 || tmat_(3,2) != 0 || tmat_(3,3) !=  7 ||
          tmat_(4,0) != -6 || tmat_(4,1) !=  0 || tmat_(4,2) != 0 || tmat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  5 -3  0  9 )\n"
                                     "(  4  0  1 -8 )\n"
                                     "(  0 -2  0  7 )\n"
                                     "( -6  0  0 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsDefault()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major isDefault() function";

      using blaze::isDefault;

      initialize();

      // isDefault with default row selection
      {
         auto rs = blaze::rows( mat_, { 0UL } );

         if( isDefault( rs(0,1) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row element: " << rs(0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( rs ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default row selection
      {
         auto rs = blaze::rows( mat_, { 1UL } );

         if( isDefault( rs(0,1) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row element: " << rs(0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( rs ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major isDefault() function";

      using blaze::isDefault;

      initialize();

      // isDefault with default row selection
      {
         auto rs = blaze::rows( tmat_, { 0UL } );

         if( isDefault( rs(0,1) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row element: " << rs(0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( rs ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default row selection
      {
         auto rs = blaze::rows( tmat_, { 1UL } );

         if( isDefault( rs(0,1) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row element: " << rs(0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( rs ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testIsSame()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function";

      // isSame with matrix and matching row selection
      {
         auto rs = blaze::rows( mat_, { 0UL, 1UL, 2UL, 3UL, 4UL } );

         if( blaze::isSame( rs, mat_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching row selection (different number of rows)
      {
         auto rs = blaze::rows( mat_, { 0UL, 1UL, 2UL, 3UL } );

         if( blaze::isSame( rs, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching row selection (different order of rows)
      {
         auto rs = blaze::rows( mat_, { 0UL, 2UL, 1UL, 3UL, 4UL } );

         if( blaze::isSame( rs, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching row selection (repeating rows)
      {
         auto rs = blaze::rows( mat_, { 0UL, 1UL, 1UL, 3UL, 4UL } );

         if( blaze::isSame( rs, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and matching row selection
      {
         auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different number of rows)
      {
         auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 1UL, 0UL, 2UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different number of columns)
      {
         auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different order of rows)
      {
         auto rs = blaze::rows( mat_, { 1UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (repeating rows)
      {
         auto rs = blaze::rows( mat_, { 1UL, 3UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different row index)
      {
         auto rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 2UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching row selections
      {
         auto rs1 = blaze::rows( mat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( mat_, { 0UL, 3UL, 1UL } );

         if( blaze::isSame( rs1, rs2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching row selections (different number of rows)
      {
         auto rs1 = blaze::rows( mat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( mat_, { 0UL, 3UL, 1UL, 2UL } );

         if( blaze::isSame( rs1, rs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching row selections (different order of rows)
      {
         auto rs1 = blaze::rows( mat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( mat_, { 0UL, 1UL, 3UL } );

         if( blaze::isSame( rs1, rs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching row selections (repeating rows)
      {
         auto rs1 = blaze::rows( mat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( mat_, { 0UL, 1UL, 1UL } );

         if( blaze::isSame( rs1, rs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function";

      // isSame with matrix and matching row selection
      {
         auto rs = blaze::rows( tmat_, { 0UL, 1UL, 2UL, 3UL, 4UL } );

         if( blaze::isSame( rs, tmat_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching row selection (different number of rows)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 1UL, 2UL, 3UL } );

         if( blaze::isSame( rs, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching row selection (different order of rows)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 2UL, 1UL, 3UL, 4UL } );

         if( blaze::isSame( rs, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching row selection (repeating rows)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 1UL, 1UL, 3UL, 4UL } );

         if( blaze::isSame( rs, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and matching row selection
      {
         auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different number of rows)
      {
         auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 1UL, 0UL, 2UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different number of columns)
      {
         auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different order of rows)
      {
         auto rs = blaze::rows( tmat_, { 1UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (repeating rows)
      {
         auto rs = blaze::rows( tmat_, { 1UL, 3UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching row selection (different row index)
      {
         auto rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 2UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching row selections
      {
         auto rs1 = blaze::rows( tmat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( tmat_, { 0UL, 3UL, 1UL } );

         if( blaze::isSame( rs1, rs2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching row selections (different number of rows)
      {
         auto rs1 = blaze::rows( tmat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( tmat_, { 0UL, 3UL, 1UL, 2UL } );

         if( blaze::isSame( rs1, rs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching row selections (different order of rows)
      {
         auto rs1 = blaze::rows( tmat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( tmat_, { 0UL, 1UL, 3UL } );

         if( blaze::isSame( rs1, rs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching row selections (repeating rows)
      {
         auto rs1 = blaze::rows( tmat_, { 0UL, 3UL, 1UL } );
         auto rs2 = blaze::rows( tmat_, { 0UL, 1UL, 1UL } );

         if( blaze::isSame( rs1, rs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First row selection:\n" << rs1 << "\n"
                << "   Second row selection:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c submatrix() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c submatrix() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testSubmatrix()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major submatrix() function";

      initialize();

      {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 2UL );

         if( sm(0,0) !=  4 || sm(0,1) != 5 ||
             sm(1,0) !=  1 || sm(1,1) != 0 ||
             sm(2,0) != -8 || sm(2,1) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4  5 )\n"
                                        "(  1  0 )\n"
                                        "( -8  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *sm.begin(1UL) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm.begin(1UL) << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 3UL, 1UL, 3UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 4UL, 3UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 4UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major submatrix() function";

      initialize();

      {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 2UL );

         if( sm(0,0) !=  4 || sm(0,1) != 5 ||
             sm(1,0) !=  1 || sm(1,1) != 0 ||
             sm(2,0) != -8 || sm(2,1) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4  5 )\n"
                                        "(  1  0 )\n"
                                        "( -8  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *sm.begin(1UL) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *sm.begin(1UL) << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 3UL, 1UL, 3UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 4UL, 3UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 4UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c row() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testRow()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major row() function";

      initialize();

      {
         auto rs   = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto row1 = row( rs, 1UL );

         if( row1[0] != 0 || row1[1] != 1 || row1[2] != 0 || row1[3] != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *row1.begin() != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *row1.begin() << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs   = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto row3 = blaze::row( rs, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major row() function";

      initialize();

      {
         auto rs   = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto row1 = row( rs, 1UL );

         if( row1[0] != 0 || row1[1] != 1 || row1[2] != 0 || row1[3] != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *row1.begin() != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *row1.begin() << "\n"
                << "   Expected result: 0\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs   = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto row3 = blaze::row( rs, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rows() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rows() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testRows()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major rows() function";

      initialize();

      {
         auto rs1 = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto rs2 = blaze::rows( rs1, { 1UL, 0UL, 2UL } );

         if( rs2(0,0) != 0 || rs2(0,1) !=  1 || rs2(0,2) != 0 || rs2(0,3) !=  0 ||
             rs2(1,0) != 0 || rs2(1,1) !=  4 || rs2(1,2) != 5 || rs2(1,3) != -6 ||
             rs2(2,0) != 7 || rs2(2,1) != -8 || rs2(2,2) != 9 || rs2(2,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n"
                << "   Expected result:\n( 0  1  0  0 )\n( 0  4  5 -6 )\n( 7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rs2.begin( 2UL ) != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs2.begin( 2UL ) << "\n"
                << "   Expected result: 7\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs1 = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto rs2 = blaze::rows( rs1, { 3UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major rows() function";

      initialize();

      {
         auto rs1 = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto rs2 = blaze::rows( rs1, { 1UL, 0UL, 2UL } );

         if( rs2(0,0) != 0 || rs2(0,1) !=  1 || rs2(0,2) != 0 || rs2(0,3) !=  0 ||
             rs2(1,0) != 0 || rs2(1,1) !=  4 || rs2(1,2) != 5 || rs2(1,3) != -6 ||
             rs2(2,0) != 7 || rs2(2,1) != -8 || rs2(2,2) != 9 || rs2(2,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n"
                << "   Expected result:\n( 0  1  0  0 )\n( 0  4  5 -6 )\n( 7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *rs2.begin( 2UL ) != 7 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs2.begin( 2UL ) << "\n"
                << "   Expected result: 7\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs1 = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto rs2 = blaze::rows( rs1, { 3UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c column() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testColumn()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major column() function";

      initialize();

      {
         auto rs   = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto col1 = blaze::column( rs, 1UL );

         if( col1[0] != 4 || col1[1] != 1 || col1[2] != -8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 4 1 -8 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *col1.begin() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *col1.begin() << "\n"
                << "   Expected result: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs   = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto col4 = blaze::column( rs, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col4 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major column() function";

      initialize();

      {
         auto rs   = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto col1 = blaze::column( rs, 1UL );

         if( col1[0] != 4 || col1[1] != 1 || col1[2] != -8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 4 1 -8 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *col1.begin() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *col1.begin() << "\n"
                << "   Expected result: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs   = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto col4 = blaze::column( rs, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col4 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c columns() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columns() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testColumns()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major columns() function";

      initialize();

      {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto cs = blaze::columns( rs, { 1UL, 0UL, 2UL } );

         if( cs(0,0) !=  4 || cs(0,1) != 0 || cs(0,2) != 5 ||
             cs(1,0) !=  1 || cs(1,1) != 0 || cs(1,2) != 0 ||
             cs(2,0) != -8 || cs(2,1) != 7 || cs(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n(  4  0  5 )\n(  1  0  0 )\n( -8  7  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *cs.begin( 2UL ) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *cs.begin( 2UL ) << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto cs = blaze::columns( rs, { 4UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major columns() function";

      initialize();

      {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto cs = blaze::columns( rs, { 1UL, 0UL, 2UL } );

         if( cs(0,0) !=  4 || cs(0,1) != 0 || cs(0,2) != 5 ||
             cs(1,0) !=  1 || cs(1,1) != 0 || cs(1,2) != 0 ||
             cs(2,0) != -8 || cs(2,1) != 7 || cs(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n(  4  0  5 )\n(  1  0  0 )\n( -8  7  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *cs.begin( 2UL ) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *cs.begin( 2UL ) << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto cs = blaze::columns( rs, { 4UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c band() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c band() function with the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseGeneralTest::testBand()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major band() function";

      initialize();

      {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto b1 = blaze::band( rs, 1L );

         if( b1[0] != 4 || b1[1] != 0 || b1[2] != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result\n: ( 4 0 10 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *b1.begin() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *b1.begin() << "\n"
                << "   Expected result: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto b4 = blaze::band( rs, 4L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b4 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( mat_, { 3UL, 1UL, 4UL } );
         auto b3 = blaze::band( rs, -3L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major band() function";

      initialize();

      {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto b1 = blaze::band( rs, 1L );

         if( b1[0] != 4 || b1[1] != 0 || b1[2] != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result\n: ( 4 0 10 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( *b1.begin() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *b1.begin() << "\n"
                << "   Expected result: 4\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto b4 = blaze::band( rs, 4L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b4 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto rs = blaze::rows( tmat_, { 3UL, 1UL, 4UL } );
         auto b3 = blaze::band( rs, -3L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member matrices to specific predetermined values.
*/
void DenseGeneralTest::initialize()
{
   // Initializing the row-major dynamic matrix
   mat_.reset();
   mat_(1,1) =  1;
   mat_(2,0) = -2;
   mat_(2,2) = -3;
   mat_(3,1) =  4;
   mat_(3,2) =  5;
   mat_(3,3) = -6;
   mat_(4,0) =  7;
   mat_(4,1) = -8;
   mat_(4,2) =  9;
   mat_(4,3) = 10;

   // Initializing the column-major dynamic matrix
   tmat_.reset();
   tmat_(1,1) =  1;
   tmat_(2,0) = -2;
   tmat_(2,2) = -3;
   tmat_(3,1) =  4;
   tmat_(3,2) =  5;
   tmat_(3,3) = -6;
   tmat_(4,0) =  7;
   tmat_(4,1) = -8;
   tmat_(4,2) =  9;
   tmat_(4,3) = 10;
}
//*************************************************************************************************

} // namespace rows

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
   std::cout << "   Running Rows dense general test (part 2)..." << std::endl;

   try
   {
      RUN_ROWS_DENSEGENERAL_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Rows dense general test (part 2):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
