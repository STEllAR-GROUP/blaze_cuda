//=================================================================================================
/*!
//  \file src/mathtest/submatrix/SparseTest2.cpp
//  \brief Source file for the Submatrix sparse test (part 2)
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
#include <blaze/math/Views.h>
#include <blazetest/mathtest/submatrix/SparseTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace submatrix {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the Submatrix sparse test.
//
// \exception std::runtime_error Operation error detected.
*/
SparseTest::SparseTest()
   : mat_ ( 5UL, 4UL )
   , tmat_( 4UL, 5UL )
{
   testScaling();
   testFunctionCall();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
   testReserve();
   testTrim();
   testSet();
   testInsert();
   testAppend();
   testErase();
   testFind();
   testLowerBound();
   testUpperBound();
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
/*!\brief Test of all Submatrix (self-)scaling operations.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all available ways to scale an instance of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testScaling()
{
   //=====================================================================================
   // Row-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M*=s) (2x3)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 2UL, 3UL );

      sm *= 3;

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 || sm(0,2) != -9 ||
          sm(1,0) !=  0 || sm(1,1) != 12 || sm(1,2) != 15 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0 -9 )\n(  0 12 15 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -6 || mat_(2,1) !=  0 || mat_(2,2) != -9 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) != 12 || mat_(3,2) != 15 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  1  0  0 )\n"
                                     "( -6  0 -9  0 )\n"
                                     "(  0 12 15 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M*=s) (3x2)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 3UL, 2UL );

      sm *= 3;

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  2UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=   0 ||
          sm(1,0) !=  0 || sm(1,1) !=  12 ||
          sm(2,0) != 21 || sm(2,1) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6   0 )\n(  0  12 )\n( 21 -24 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=   0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=   1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -6 || mat_(2,1) !=   0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  12 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) != 21 || mat_(4,1) != -24 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0  0  0 )\n"
                                     "(  0   1  0  0 )\n"
                                     "( -6   0 -3  0 )\n"
                                     "(  0  12  5 -6 )\n"
                                     "( 21 -24  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M*s) (2x3)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 2UL, 3UL );

      sm = sm * 3;

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 || sm(0,2) != -9 ||
          sm(1,0) !=  0 || sm(1,1) != 12 || sm(1,2) != 15 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0 -9 )\n(  0 12 15 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -6 || mat_(2,1) !=  0 || mat_(2,2) != -9 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) != 12 || mat_(3,2) != 15 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  1  0  0 )\n"
                                     "( -6  0 -9  0 )\n"
                                     "(  0 12 15 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M*s) (3x2)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 3UL, 2UL );

      sm = sm * 3;

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  2UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=   0 ||
          sm(1,0) !=  0 || sm(1,1) !=  12 ||
          sm(2,0) != 21 || sm(2,1) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6   0 )\n(  0  12 )\n( 21 -24 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=   0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=   1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -6 || mat_(2,1) !=   0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  12 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) != 21 || mat_(4,1) != -24 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0  0  0 )\n"
                                     "(  0   1  0  0 )\n"
                                     "( -6   0 -3  0 )\n"
                                     "(  0  12  5 -6 )\n"
                                     "( 21 -24  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=s*M) (2x3)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 2UL, 3UL );

      sm = 3 * sm;

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 || sm(0,2) != -9 ||
          sm(1,0) !=  0 || sm(1,1) != 12 || sm(1,2) != 15 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0 -9 )\n(  0 12 15 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -6 || mat_(2,1) !=  0 || mat_(2,2) != -9 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) != 12 || mat_(3,2) != 15 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  1  0  0 )\n"
                                     "( -6  0 -9  0 )\n"
                                     "(  0 12 15 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=s*M) (3x2)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 3UL, 2UL );

      sm = 3 * sm;

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  2UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=   0 ||
          sm(1,0) !=  0 || sm(1,1) !=  12 ||
          sm(2,0) != 21 || sm(2,1) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6   0 )\n(  0  12 )\n( 21 -24 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=   0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=   1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -6 || mat_(2,1) !=   0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  12 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) != 21 || mat_(4,1) != -24 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0  0  0 )\n"
                                     "(  0   1  0  0 )\n"
                                     "( -6   0 -3  0 )\n"
                                     "(  0  12  5 -6 )\n"
                                     "( 21 -24  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M/=s) (2x3)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 2UL, 3UL );

      sm /= 0.5;

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -4 || sm(0,1) != 0 || sm(0,2) != -6 ||
          sm(1,0) !=  0 || sm(1,1) != 8 || sm(1,2) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -4  0 -6 )\n(  0  8 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -4 || mat_(2,1) !=  0 || mat_(2,2) != -6 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) != 10 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  1  0  0 )\n"
                                     "( -4  0 -6  0 )\n"
                                     "(  0  8 10 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M/=s) (3x2)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 3UL, 2UL );

      sm /= 0.5;

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  2UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -4 || sm(0,1) !=   0 ||
          sm(1,0) !=  0 || sm(1,1) !=   8 ||
          sm(2,0) != 14 || sm(2,1) != -16 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -4   0 )\n(  0   8 )\n( 14 -16 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=   0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=   1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -4 || mat_(2,1) !=   0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=   8 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) != 14 || mat_(4,1) != -16 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0  0  0 )\n"
                                     "(  0   1  0  0 )\n"
                                     "( -4   0 -3  0 )\n"
                                     "(  0   8  5 -6 )\n"
                                     "( 14 -16  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major self-scaling (M=M/s)
   //=====================================================================================

   {
      test_ = "Row-major self-scaling (M=M/s) (2x3)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 2UL, 3UL );

      sm = sm / 0.5;

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -4 || sm(0,1) != 0 || sm(0,2) != -6 ||
          sm(1,0) !=  0 || sm(1,1) != 8 || sm(1,2) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -4  0 -6 )\n(  0  8 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -4 || mat_(2,1) !=  0 || mat_(2,2) != -6 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) != 10 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  1  0  0 )\n"
                                     "( -4  0 -6  0 )\n"
                                     "(  0  8 10 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-scaling (M=M/s) (3x2)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 0UL, 3UL, 2UL );

      sm = sm / 0.5;

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  2UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != -4 || sm(0,1) !=   0 ||
          sm(1,0) !=  0 || sm(1,1) !=   8 ||
          sm(2,0) != 14 || sm(2,1) != -16 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -4   0 )\n(  0   8 )\n( 14 -16 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=   0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=   1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) != -4 || mat_(2,1) !=   0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=   8 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) != 14 || mat_(4,1) != -16 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0   0  0  0 )\n"
                                     "(  0   1  0  0 )\n"
                                     "( -4   0 -3  0 )\n"
                                     "(  0   8  5 -6 )\n"
                                     "( 14 -16  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major Submatrix::scale()
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::scale()";

      initialize();

      // Initialization check
      SMT sm = blaze::submatrix( mat_, 2UL, 1UL, 2UL, 2UL );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 3UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 2UL );

      if( sm(0,0) != 0 || sm(0,1) != -3 ||
          sm(1,0) != 4 || sm(1,1) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -3 )\n( 4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Integral scaling of the matrix
      sm.scale( 2 );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 3UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 2UL );

      if( sm(0,0) != 0 || sm(0,1) != -6 ||
          sm(1,0) != 8 || sm(1,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -6 )\n( 8 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the matrix
      sm.scale( 0.5 );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 3UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 2UL );

      if( sm(0,0) != 0 || sm(0,1) != -3 ||
          sm(1,0) != 4 || sm(1,1) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -3 )\n( 4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M*=s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M*=s) (3x2)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 3UL, 2UL );

      sm *= 3;

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 ||
          sm(1,0) !=  0 || sm(1,1) != 12 ||
          sm(2,0) != -9 || sm(2,1) != 15 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0 )\n(  0 12 )\n( -9 15 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -6 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) != 12 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -9 || tmat_(2,3) != 15 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -6  0  7 )\n"
                                     "( 0  1  0 12 -8 )\n"
                                     "( 0  0 -9 15  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-scaling (M*=s) (2x3)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 2UL, 3UL );

      sm *= 3;

      checkRows    ( sm   ,  2UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 || sm(0,2) !=  21 ||
          sm(1,0) !=  0 || sm(1,1) != 12 || sm(1,2) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0  21 )\n(  0 12 -24 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -6 || tmat_(0,3) !=  0 || tmat_(0,4) !=  21 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) != 12 || tmat_(1,4) != -24 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=   9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -6  0  21 )\n"
                                     "( 0  1  0 12 -24 )\n"
                                     "( 0  0 -3  5   9 )\n"
                                     "( 0  0  0 -6  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=M*s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M=M*s) (3x2)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 3UL, 2UL );

      sm = sm * 3;

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 ||
          sm(1,0) !=  0 || sm(1,1) != 12 ||
          sm(2,0) != -9 || sm(2,1) != 15 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0 )\n(  0 12 )\n( -9 15 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -6 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) != 12 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -9 || tmat_(2,3) != 15 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -6  0  7 )\n"
                                     "( 0  1  0 12 -8 )\n"
                                     "( 0  0 -9 15  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-scaling (M=M*s) (2x3)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 2UL, 3UL );

      sm = sm * 3;

      checkRows    ( sm   ,  2UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 || sm(0,2) !=  21 ||
          sm(1,0) !=  0 || sm(1,1) != 12 || sm(1,2) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0  21 )\n(  0 12 -24 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -6 || tmat_(0,3) !=  0 || tmat_(0,4) !=  21 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) != 12 || tmat_(1,4) != -24 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=   9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -6  0  21 )\n"
                                     "( 0  1  0 12 -24 )\n"
                                     "( 0  0 -3  5   9 )\n"
                                     "( 0  0  0 -6  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M=s*M)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M=s*M) (3x2)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 3UL, 2UL );

      sm = 3 * sm;

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 ||
          sm(1,0) !=  0 || sm(1,1) != 12 ||
          sm(2,0) != -9 || sm(2,1) != 15 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0 )\n(  0 12 )\n( -9 15 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -6 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) != 12 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -9 || tmat_(2,3) != 15 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -6  0  7 )\n"
                                     "( 0  1  0 12 -8 )\n"
                                     "( 0  0 -9 15  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-scaling (M=s*M) (2x3)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 2UL, 3UL );

      sm = 3 * sm;

      checkRows    ( sm   ,  2UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -6 || sm(0,1) !=  0 || sm(0,2) !=  21 ||
          sm(1,0) !=  0 || sm(1,1) != 12 || sm(1,2) != -24 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -6  0  21 )\n(  0 12 -24 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -6 || tmat_(0,3) !=  0 || tmat_(0,4) !=  21 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) != 12 || tmat_(1,4) != -24 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=   9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -6  0  21 )\n"
                                     "( 0  1  0 12 -24 )\n"
                                     "( 0  0 -3  5   9 )\n"
                                     "( 0  0  0 -6  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major self-scaling (M/=s)
   //=====================================================================================

   {
      test_ = "Column-major self-scaling (M/=s) (3x2)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 3UL, 2UL );

      sm /= 0.5;

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -4 || sm(0,1) !=  0 ||
          sm(1,0) !=  0 || sm(1,1) !=  8 ||
          sm(2,0) != -6 || sm(2,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -4  0 )\n(  0  8 )\n( -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -4 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -6 || tmat_(2,3) != 10 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -4  0  7 )\n"
                                     "( 0  1  0  8 -8 )\n"
                                     "( 0  0 -6 10  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-scaling (M/=s) (2x3)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 2UL, 2UL, 3UL );

      sm /= 0.5;

      checkRows    ( sm   ,  2UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) != -4 || sm(0,1) != 0 || sm(0,2) !=  14 ||
          sm(1,0) !=  0 || sm(1,1) != 8 || sm(1,2) != -16 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( -4  0  14 )\n(  0  8 -16 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -4 || tmat_(0,3) !=  0 || tmat_(0,4) !=  14 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 || tmat_(1,4) != -16 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=   9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) !=  10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Failed self-scaling operation\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -4  0  14 )\n"
                                     "( 0  1  0  8 -16 )\n"
                                     "( 0  0 -3  5   9 )\n"
                                     "( 0  0  0 -6  10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major Submatrix::scale()
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::scale()";

      initialize();

      // Initialization check
      OSMT sm = blaze::submatrix( tmat_, 1UL, 2UL, 2UL, 2UL );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 3UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 2UL );

      if( sm(0,0) !=  0 || sm(0,1) != 4 ||
          sm(1,0) != -3 || sm(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 4 )\n( -3 5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Integral scaling of the matrix
      sm.scale( 2 );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 3UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 2UL );

      if( sm(0,0) !=  0 || sm(0,1) !=  8 ||
          sm(1,0) != -6 || sm(1,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Integral scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  8 )\n( -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Floating point scaling of the matrix
      sm.scale( 0.5 );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 3UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 2UL );

      if( sm(0,0) !=  0 || sm(0,1) != 4 ||
          sm(1,0) != -3 || sm(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Floating point scale operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 4 )\n( -3 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Submatrix function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the Submatrix specialization. In case an error is detected, a \a std::runtime_error
// exception is thrown.
*/
void SparseTest::testFunctionCall()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::operator()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );

      // Assignment to the element (1,0)
      {
         sm(1,0) = 9;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  5UL );
         checkNonZeros( sm  ,  0UL, 1UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 2UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( sm(0,0) != 1 || sm(0,1) !=  0 ||
             sm(1,0) != 9 || sm(1,1) != -3 ||
             sm(2,0) != 4 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 1  0 )\n( 9 -3 )\n( 4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  4 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
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

      // Assignment to the element (2,0)
      {
         sm(2,0) = 0;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  4UL );
         checkNonZeros( sm  ,  0UL, 1UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 1UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 10UL );

         if( sm(0,0) != 1 || sm(0,1) !=  0 ||
             sm(1,0) != 9 || sm(1,1) != -3 ||
             sm(2,0) != 0 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 1  0 )\n( 9 -3 )\n( 0  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
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

      // Assignment to the element (1,1)
      {
         sm(1,1) = 11;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  4UL );
         checkNonZeros( sm  ,  0UL, 1UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 1UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 10UL );

         if( sm(0,0) != 1 || sm(0,1) !=  0 ||
             sm(1,0) != 9 || sm(1,1) != 11 ||
             sm(2,0) != 0 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 1  0 )\n( 9 11 )\n( 0  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  9 || mat_(2,2) != 11 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
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

      // Addition assignment to the element (0,0)
      {
         sm(0,0) += 3;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  4UL );
         checkNonZeros( sm  ,  0UL, 1UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 1UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 10UL );

         if( sm(0,0) != 4 || sm(0,1) !=  0 ||
             sm(1,0) != 9 || sm(1,1) != 11 ||
             sm(2,0) != 0 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4  0 )\n( 9 11 )\n( 0  5 )\n";
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

      // Subtraction assignment to the element (0,1)
      {
         sm(0,1) -= 6;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  5UL );
         checkNonZeros( sm  ,  0UL, 2UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 1UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( sm(0,0) != 4 || sm(0,1) != -6 ||
             sm(1,0) != 9 || sm(1,1) != 11 ||
             sm(2,0) != 0 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4 -6 )\n( 9 11 )\n( 0  5 )\n";
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

      // Multiplication assignment to the element (1,1)
      {
         sm(1,1) *= 2;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  5UL );
         checkNonZeros( sm  ,  0UL, 2UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 1UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( sm(0,0) != 4 || sm(0,1) != -6 ||
             sm(1,0) != 9 || sm(1,1) != 22 ||
             sm(2,0) != 0 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4 -6 )\n( 9 22 )\n( 0  5 )\n";
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

      // Division assignment to the element (1,1)
      {
         sm(1,1) /= 2;

         checkRows    ( sm  ,  3UL );
         checkColumns ( sm  ,  2UL );
         checkNonZeros( sm  ,  5UL );
         checkNonZeros( sm  ,  0UL, 2UL );
         checkNonZeros( sm  ,  1UL, 2UL );
         checkNonZeros( sm  ,  2UL, 1UL );
         checkRows    ( mat_,  5UL );
         checkColumns ( mat_,  4UL );
         checkNonZeros( mat_, 11UL );

         if( sm(0,0) != 4 || sm(0,1) != -6 ||
             sm(1,0) != 9 || sm(1,1) != 11 ||
             sm(2,0) != 0 || sm(2,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4 -6 )\n( 9 11 )\n( 0  5 )\n";
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
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::operator()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );

      // Assignment to the element (0,1)
      {
         sm(0,1) = 9;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  5UL );
         checkNonZeros( sm   ,  0UL, 1UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 2UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 11UL );

         if( sm(0,0) != 1 || sm(0,1) !=  9 || sm(0,2) != 4 ||
             sm(1,0) != 0 || sm(1,1) != -3 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 1  9 4 )\n( 0 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  9 || tmat_(1,3) !=  4 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  1  9  4 -8 )\n"
                                        "( 0  0 -3  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (0,2)
      {
         sm(0,2) = 0;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  4UL );
         checkNonZeros( sm   ,  0UL, 1UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 1UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 10UL );

         if( sm(0,0) != 1 || sm(0,1) !=  9 || sm(0,2) != 0 ||
             sm(1,0) != 0 || sm(1,1) != -3 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 1  9 0 )\n( 0 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  9 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  1  9  0 -8 )\n"
                                        "( 0  0 -3  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,1)
      {
         sm(1,1) = 11;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  4UL );
         checkNonZeros( sm   ,  0UL, 1UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 1UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 10UL );

         if( sm(0,0) != 1 || sm(0,1) !=  9 || sm(0,2) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 11 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 1 11 0 )\n( 0 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  9 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != 11 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  1  9  0 -8 )\n"
                                        "( 0  0 11  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (0,0)
      {
         sm(0,0) += 3;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  4UL );
         checkNonZeros( sm   ,  0UL, 1UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 1UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 10UL );

         if( sm(0,0) != 4 || sm(0,1) !=  9 || sm(0,2) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 11 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4 11 0 )\n( 0 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 4 || tmat_(1,2) !=  9 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != 11 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  4  9  0 -8 )\n"
                                        "( 0  0 11  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (1,0)
      {
         sm(1,0) -= 6;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  5UL );
         checkNonZeros( sm   ,  0UL, 2UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 1UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 11UL );

         if( sm(0,0) !=  4 || sm(0,1) !=  9 || sm(0,2) != 0 ||
             sm(1,0) != -6 || sm(1,1) != 11 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 11 0 )\n( -6 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  4 || tmat_(1,2) !=  9 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != -6 || tmat_(2,2) != 11 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  4  9  0 -8 )\n"
                                        "( 0 -6 11  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,1)
      {
         sm(1,1) *= 2;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  5UL );
         checkNonZeros( sm   ,  0UL, 2UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 1UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 11UL );

         if( sm(0,0) !=  4 || sm(0,1) !=  9 || sm(0,2) != 0 ||
             sm(1,0) != -6 || sm(1,1) != 22 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 22 0 )\n( -6 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  4 || tmat_(1,2) !=  9 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != -6 || tmat_(2,2) != 22 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  4  9  0 -8 )\n"
                                        "( 0 -6 22  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (1,1)
      {
         sm(1,1) /= 2;

         checkRows    ( sm   ,  2UL );
         checkColumns ( sm   ,  3UL );
         checkNonZeros( sm   ,  5UL );
         checkNonZeros( sm   ,  0UL, 2UL );
         checkNonZeros( sm   ,  1UL, 2UL );
         checkNonZeros( sm   ,  2UL, 1UL );
         checkRows    ( tmat_,  4UL );
         checkColumns ( tmat_,  5UL );
         checkNonZeros( tmat_, 11UL );

         if( sm(0,0) !=  4 || sm(0,1) !=  9 || sm(0,2) != 0 ||
             sm(1,0) != -6 || sm(1,1) != 11 || sm(1,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 11 0 )\n( -6 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  4 || tmat_(1,2) !=  9 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != -6 || tmat_(2,2) != 11 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  4  9  0 -8 )\n"
                                        "( 0 -6 11  5  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the Submatrix iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testIterator()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 3UL );

      // Testing the Iterator default constructor
      {
         test_ = "Row-major Iterator default constructor";

         SMT::Iterator it{};

         if( it != SMT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         SMT::ConstIterator it{};

         if( it != SMT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         SMT::ConstIterator it( begin( sm, 1UL ) );

         if( it == end( sm, 1UL ) || it->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th row via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         const ptrdiff_t number( end( sm, 0UL ) - begin( sm, 0UL ) );

         if( number != 1L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         const ptrdiff_t number( cend( sm, 1UL ) - cbegin( sm, 1UL ) );

         if( number != 2L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 2\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Row-major read-only access via ConstIterator";

         SMT::ConstIterator it ( cbegin( sm, 2UL ) );
         SMT::ConstIterator end( cend( sm, 2UL ) );

         if( it == end || it->value() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || it->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it != cend( sm, 2UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Row-major assignment via Iterator";

         int value = 8;

         for( SMT::Iterator it=begin( sm, 2UL ); it!=end( sm, 2UL ); ++it ) {
            *it = value++;
         }

         if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
             sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -3 ||
             sm(2,0) !=  0 || sm(2,1) != 8 || sm(2,2) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0  1  0 )\n( -2  0 -3 )\n(  0  8  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) !=  9 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  8  9 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         int value = 4;

         for( SMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it += value++;
         }

         if( sm(0,0) != 0 || sm(0,1) != 1 || sm(0,2) != 0 ||
             sm(1,0) != 2 || sm(1,1) != 0 || sm(1,2) != 2 ||
             sm(2,0) != 0 || sm(2,1) != 8 || sm(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 1 0 )\n( 2 0 2 )\n( 0 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
             mat_(1,0) != 0 || mat_(1,1) !=  1 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
             mat_(2,0) != 2 || mat_(2,1) !=  0 || mat_(2,2) != 2 || mat_(2,3) !=  0 ||
             mat_(3,0) != 0 || mat_(3,1) !=  8 || mat_(3,2) != 9 || mat_(3,3) != -6 ||
             mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "(  2  0  2  0 )\n"
                                        "(  0  8  9 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         int value = 4;

         for( SMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it -= value++;
         }

         if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
             sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -3 ||
             sm(2,0) !=  0 || sm(2,1) != 8 || sm(2,2) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0  1  0 )\n( -2  0 -3 )\n(  0  8  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) !=  9 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -3  0 )\n"
                                        "(  0  8  9 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         int value = 1;

         for( SMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it *= value++;
         }

         if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
             sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -6 ||
             sm(2,0) !=  0 || sm(2,1) != 8 || sm(2,2) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0  1  0 )\n( -2  0 -6 )\n(  0  8  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -2 || mat_(2,1) !=  0 || mat_(2,2) != -6 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) !=  9 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -2  0 -6  0 )\n"
                                        "(  0  8  9 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         for( SMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it /= 2;
         }

         if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
             sm(1,0) != -1 || sm(1,1) != 0 || sm(1,2) != -3 ||
             sm(2,0) !=  0 || sm(2,1) != 8 || sm(2,2) !=  9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0  1  0 )\n( -1  0 -3 )\n(  0  8  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) !=  0 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
             mat_(2,0) != -1 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
             mat_(3,0) !=  0 || mat_(3,1) !=  8 || mat_(3,2) !=  9 || mat_(3,3) != -6 ||
             mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  0  0  0  0 )\n"
                                        "(  0  1  0  0 )\n"
                                        "( -1  0 -3  0 )\n"
                                        "(  0  8  9 -6 )\n"
                                        "(  7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 3UL, 3UL );

      // Testing the Iterator default constructor
      {
         test_ = "Row-major Iterator default constructor";

         OSMT::Iterator it{};

         if( it != OSMT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         OSMT::ConstIterator it{};

         if( it != OSMT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Column-major Iterator/ConstIterator conversion";

         OSMT::ConstIterator it( begin( sm, 1UL ) );

         if( it == end( sm, 1UL ) || it->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 0th column via Iterator (end-begin)
      {
         test_ = "Column-major Iterator subtraction (end-begin)";

         const ptrdiff_t number( end( sm, 0UL ) - begin( sm, 0UL ) );

         if( number != 1L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via ConstIterator (end-begin)
      {
         test_ = "Column-major ConstIterator subtraction (end-begin)";

         const ptrdiff_t number( cend( sm, 1UL ) - cbegin( sm, 1UL ) );

         if( number != 2L ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid number of elements detected\n"
                << " Details:\n"
                << "   Number of elements         : " << number << "\n"
                << "   Expected number of elements: 2\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing read-only access via ConstIterator
      {
         test_ = "Column-major read-only access via ConstIterator";

         OSMT::ConstIterator it ( cbegin( sm, 2UL ) );
         OSMT::ConstIterator end( cend( sm, 2UL ) );

         if( it == end || it->value() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || it->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it != cend( sm, 2UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Column-major assignment via Iterator";

         int value = 8;

         for( OSMT::Iterator it=begin( sm, 2UL ); it!=end( sm, 2UL ); ++it ) {
            *it = value++;
         }

         if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
             sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 8 ||
             sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 -2  0 )\n( 1  0  8 )\n( 0 -3  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  9 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  1  0  8 -8 )\n"
                                        "( 0  0 -3  9  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Column-major addition assignment via Iterator";

         int value = 4;

         for( OSMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it += value++;
         }

         if( sm(0,0) != 0 || sm(0,1) != 2 || sm(0,2) != 0 ||
             sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 8 ||
             sm(2,0) != 0 || sm(2,1) != 2 || sm(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 2 0 )\n( 1 0 8 )\n( 0 2 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  1 || tmat_(1,2) != 0 || tmat_(1,3) !=  8 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 2 || tmat_(2,3) !=  9 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Addition assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  2  0  7 )\n"
                                        "( 0  1  0  8 -8 )\n"
                                        "( 0  0  2  9  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Column-major subtraction assignment via Iterator";

         int value = 4;

         for( OSMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it -= value++;
         }

         if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
             sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 8 ||
             sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 -2  0 )\n( 1  0  8 )\n( 0 -3  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  9 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subtraction assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  1  0  8 -8 )\n"
                                        "( 0  0 -3  9  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Column-major multiplication assignment via Iterator";

         int value = 1;

         for( OSMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it *= value++;
         }

         if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
             sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 8 ||
             sm(2,0) != 0 || sm(2,1) != -6 || sm(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 -2  0 )\n( 1  0  8 )\n( 0 -6  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != -6 || tmat_(2,3) !=  9 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Multiplication assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                        "( 0  1  0  8 -8 )\n"
                                        "( 0  0 -6  9  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Column-major division assignment via Iterator";

         for( OSMT::Iterator it=begin( sm, 1UL ); it!=end( sm, 1UL ); ++it ) {
            *it /= 2;
         }

         if( sm(0,0) != 0 || sm(0,1) != -1 || sm(0,2) != 0 ||
             sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 8 ||
             sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 -1  0 )\n( 1  0  8 )\n( 0 -3  9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != -1 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 || tmat_(1,4) != -8 ||
             tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != -3 || tmat_(2,3) !=  9 || tmat_(2,4) !=  9 ||
             tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Division assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0 -1  0  7 )\n"
                                        "( 0  1  0  8 -8 )\n"
                                        "( 0  0 -3  9  9 )\n"
                                        "( 0  0  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c nonZeros() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testNonZeros()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::nonZeros()";

      initialize();

      // Initialization check
      SMT sm = blaze::submatrix( mat_, 1UL, 1UL, 2UL, 3UL );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 3UL );
      checkNonZeros( sm, 2UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 1UL );

      if( sm(0,0) != 1 || sm(0,1) !=  0 || sm(0,2) != 0 ||
          sm(1,0) != 0 || sm(1,1) != -3 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 1  0 0 )\n( 0 -3 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the sparse submatrix
      sm(1,1) = 0;

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 3UL );
      checkNonZeros( sm, 1UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 0UL );

      if( sm(0,0) != 1 || sm(0,1) != 0 || sm(0,2) != 0 ||
          sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the sparse matrix
      mat_(2,3) = 5;

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 3UL );
      checkNonZeros( sm, 2UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 1UL );

      if( sm(0,0) != 1 || sm(0,1) != 0 || sm(0,2) != 0 ||
          sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 0 0 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::nonZeros()";

      initialize();

      // Initialization check
      OSMT sm = blaze::submatrix( tmat_, 1UL, 1UL, 3UL, 2UL );

      checkRows    ( sm, 3UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 2UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 1UL );

      if( sm(0,0) != 1 || sm(0,1) !=  0 ||
          sm(1,0) != 0 || sm(1,1) != -3 ||
          sm(2,0) != 0 || sm(2,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 1  0 )\n( 0 -3 )\n( 0  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the sparse submatrix
      sm(1,1) = 0;

      checkRows    ( sm, 3UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 1UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 0UL );

      if( sm(0,0) != 1 || sm(0,1) != 0 ||
          sm(1,0) != 0 || sm(1,1) != 0 ||
          sm(2,0) != 0 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 1 0 )\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the sparse matrix
      tmat_(3,2) = 5;

      checkRows    ( sm, 3UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 2UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 1UL );

      if( sm(0,0) != 1 || sm(0,1) != 0 ||
          sm(1,0) != 0 || sm(1,1) != 0 ||
          sm(2,0) != 0 || sm(2,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 1 0 )\n( 0 0 )\n( 0 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testReset()
{
   using blaze::reset;
   using blaze::isDefault;


   //=====================================================================================
   // Row-major single element reset
   //=====================================================================================

   {
      test_ = "Row-major reset() function";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL );

      reset( sm(0,1) );

      checkRows    ( sm  , 3UL );
      checkColumns ( sm  , 2UL );
      checkNonZeros( sm  , 2UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( !isDefault( sm(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 0 )\n( -2 0 )\n(  0 4 )\n";
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
      test_ = "Row-major Submatrix::reset() (lvalue)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL );

      reset( sm );

      checkRows    ( sm  , 3UL );
      checkColumns ( sm  , 2UL );
      checkNonZeros( sm  , 0UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 7UL );

      if( !isDefault( sm ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  0 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) !=  0 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0 -3  0 )\n"
                                     "(  0  0  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Submatrix::reset() (rvalue)";

      initialize();

      reset( blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL ) );

      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 7UL );

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  0 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) !=  0 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0 -3  0 )\n"
                                     "(  0  0  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major row-wise reset
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::reset( size_t )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL );

      // Resetting the 0th row
      {
         reset( sm, 0UL );

         checkRows    ( sm  , 3UL );
         checkColumns ( sm  , 2UL );
         checkNonZeros( sm  , 2UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 9UL );

         if( sm(0,0) !=  0 || sm(0,1) != 0 ||
             sm(1,0) != -2 || sm(1,1) != 0 ||
             sm(2,0) !=  0 || sm(2,1) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 0th row failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0 0 )\n( -2 0 )\n(  0 4 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st row
      {
         reset( sm, 1UL );

         checkRows    ( sm  , 3UL );
         checkColumns ( sm  , 2UL );
         checkNonZeros( sm  , 1UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 8UL );

         if( sm(0,0) != 0 || sm(0,1) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 ||
             sm(2,0) != 0 || sm(2,1) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 4 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 2nd row
      {
         reset( sm, 2UL );

         checkRows    ( sm  , 3UL );
         checkColumns ( sm  , 2UL );
         checkNonZeros( sm  , 0UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( sm(0,0) != 0 || sm(0,1) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 ||
             sm(2,0) != 0 || sm(2,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 2nd row failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major single element reset
   //=====================================================================================

   {
      test_ = "Column-major reset() function";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL );

      reset( sm(1,0) );

      checkRows    ( sm   , 2UL );
      checkColumns ( sm   , 3UL );
      checkNonZeros( sm   , 2UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 9UL );

      if( !isDefault( sm(1,0) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -2 0 )\n( 0  0 4 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  4 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                     "( 0  0  0  4 -8 )\n"
                                     "( 0  0 -3  5  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major reset
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::reset() (lvalue)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL );

      reset( sm );

      checkRows    ( sm   , 2UL );
      checkColumns ( sm   , 3UL );
      checkNonZeros( sm   , 0UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 7UL );

      if( !isDefault( sm ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0  7 )\n"
                                     "( 0  0  0  0 -8 )\n"
                                     "( 0  0 -3  5  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major Submatrix::reset() (rvalue)";

      initialize();

      reset( blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL ) );

      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 7UL );

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0  7 )\n"
                                     "( 0  0  0  0 -8 )\n"
                                     "( 0  0 -3  5  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major row-wise reset
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::reset( size_t )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL );

      // Resetting the 0th column
      {
         reset( sm, 0UL );

         checkRows    ( sm   , 2UL );
         checkColumns ( sm   , 3UL );
         checkNonZeros( sm   , 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 9UL );

         if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
             sm(1,0) != 0 || sm(1,1) !=  0 || sm(1,2) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 -2  0 )\n( 0  0  4 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 1st column
      {
         reset( sm, 1UL );

         checkRows    ( sm   , 2UL );
         checkColumns ( sm   , 3UL );
         checkNonZeros( sm   , 1UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 8UL );

         if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 1st column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 )\n( 0 0 4 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Resetting the 2nd column
      {
         reset( sm, 2UL );

         checkRows    ( sm   , 2UL );
         checkColumns ( sm   , 3UL );
         checkNonZeros( sm   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 7UL );

         if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Reset operation of 2nd column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c clear() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testClear()
{
   using blaze::clear;
   using blaze::isDefault;


   //=====================================================================================
   // Row-major single element clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL );

      clear( sm(0,1) );

      checkRows    ( sm  , 3UL );
      checkColumns ( sm  , 2UL );
      checkNonZeros( sm  , 2UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( !isDefault( sm(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 0 )\n( -2 0 )\n(  0 4 )\n";
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
      test_ = "Row-major clear() function (lvalue)";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL );

      clear( sm );

      checkRows    ( sm  , 3UL );
      checkColumns ( sm  , 2UL );
      checkNonZeros( sm  , 0UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 7UL );

      if( !isDefault( sm ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  0 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) !=  0 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0 -3  0 )\n"
                                     "(  0  0  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major clear() function (rvalue)";

      initialize();

      clear( blaze::submatrix( mat_, 1UL, 0UL, 3UL, 2UL ) );

      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 7UL );

      if( mat_(0,0) !=  0 || mat_(0,1) !=  0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
          mat_(1,0) !=  0 || mat_(1,1) !=  0 || mat_(1,2) !=  0 || mat_(1,3) !=  0 ||
          mat_(2,0) !=  0 || mat_(2,1) !=  0 || mat_(2,2) != -3 || mat_(2,3) !=  0 ||
          mat_(3,0) !=  0 || mat_(3,1) !=  0 || mat_(3,2) !=  5 || mat_(3,3) != -6 ||
          mat_(4,0) !=  7 || mat_(4,1) != -8 || mat_(4,2) !=  9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0  0  0  0 )\n"
                                     "(  0  0 -3  0 )\n"
                                     "(  0  0  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major single element clear
   //=====================================================================================

   {
      test_ = "Column-major clear() function";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL );

      clear( sm(1,0) );

      checkRows    ( sm   , 2UL );
      checkColumns ( sm   , 3UL );
      checkNonZeros( sm   , 2UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 9UL );

      if( !isDefault( sm(1,0) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -2 0 )\n( 0  0 4 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != -2 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  4 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0 -2  0  7 )\n"
                                     "( 0  0  0  4 -8 )\n"
                                     "( 0  0 -3  5  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major clear
   //=====================================================================================

   {
      test_ = "Column-major clear() function (lvalue)";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL );

      clear( sm );

      checkRows    ( sm   , 2UL );
      checkColumns ( sm   , 3UL );
      checkNonZeros( sm   , 0UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 7UL );

      if( !isDefault( sm ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0  7 )\n"
                                     "( 0  0  0  0 -8 )\n"
                                     "( 0  0 -3  5  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major clear() function (rvalue)";

      initialize();

      clear( blaze::submatrix( tmat_, 0UL, 1UL, 2UL, 3UL ) );

      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 7UL );

      if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != 0 || tmat_(1,2) !=  0 || tmat_(1,3) !=  0 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != -3 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0  7 )\n"
                                     "( 0  0  0  0 -8 )\n"
                                     "( 0  0 -3  5  9 )\n"
                                     "( 0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reserve() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reserve() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testReserve()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::reserve()";

      MT mat( 3UL, 20UL );

      SMT sm = blaze::submatrix( mat, 1UL, 0UL, 1UL, 20UL );

      // Increasing the capacity of the matrix
      sm.reserve( 10UL );

      checkRows    ( sm,  1UL );
      checkColumns ( sm, 20UL );
      checkCapacity( sm, 10UL );
      checkNonZeros( sm,  0UL );

      // Further increasing the capacity of the matrix
      sm.reserve( 20UL );

      checkRows    ( sm,  1UL );
      checkColumns ( sm, 20UL );
      checkCapacity( sm, 20UL );
      checkNonZeros( sm,  0UL );
   }

   {
      test_ = "Row-major Submatrix::reserve( size_t )";

      MT mat( 3UL, 20UL );

      SMT sm = blaze::submatrix( mat, 1UL, 0UL, 1UL, 20UL );

      // Increasing the capacity of the row
      sm.reserve( 0UL, 10UL );

      checkRows    ( sm,  1UL );
      checkColumns ( sm, 20UL );
      checkCapacity( sm, 10UL );
      checkNonZeros( sm,  0UL );

      // Further increasing the capacity of the row
      sm.reserve( 0UL, 15UL );

      checkRows    ( sm,  1UL );
      checkColumns ( sm, 20UL );
      checkCapacity( sm, 15UL );
      checkNonZeros( sm,  0UL );
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::reserve()";

      OMT mat( 3UL, 20UL );

      OSMT sm = blaze::submatrix( mat, 1UL, 0UL, 1UL, 20UL );

      // Increasing the capacity of the matrix
      sm.reserve( 10UL );

      checkRows    ( sm,  1UL );
      checkColumns ( sm, 20UL );
      checkCapacity( sm, 10UL );
      checkNonZeros( sm,  0UL );

      // Further increasing the capacity of the matrix
      sm.reserve( 20UL );

      checkRows    ( sm,  1UL );
      checkColumns ( sm, 20UL );
      checkCapacity( sm, 20UL );
      checkNonZeros( sm,  0UL );
   }

   {
      test_ = "Columnt-major Submatrix::reserve( size_t )";

      OMT mat( 20UL, 3UL );

      OSMT sm = blaze::submatrix( mat, 0UL, 1UL, 20UL, 1UL );

      // Increasing the capacity of the column
      sm.reserve( 0UL, 10UL );

      checkRows    ( sm, 20UL );
      checkColumns ( sm,  1UL );
      checkCapacity( sm, 10UL );
      checkNonZeros( sm,  0UL );

      // Further increasing the capacity of the column
      sm.reserve( 0UL, 15UL );

      checkRows    ( sm, 20UL );
      checkColumns ( sm,  1UL );
      checkCapacity( sm, 15UL );
      checkNonZeros( sm,  0UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c trim() member functions of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c trim() member functions of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testTrim()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::trim()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 1UL, 2UL, 3UL );

      // Increasing the row capacity of the matrix
      sm.reserve( 0UL, 10UL );
      sm.reserve( 1UL, 20UL );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkCapacity( sm  , 30UL );
      checkCapacity( sm  ,  0UL, 10UL );
      checkCapacity( sm  ,  1UL, 20UL );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, 10UL );
      checkCapacity( mat_,  3UL, 20UL );

      // Trimming the matrix
      sm.trim();

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkCapacity( sm  , 30UL );
      checkCapacity( sm  ,  0UL, sm.nonZeros( 0UL ) );
      checkCapacity( sm  ,  1UL, sm.nonZeros( 1UL ) );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, mat_.nonZeros( 2UL ) );
      checkCapacity( mat_,  3UL, mat_.nonZeros( 3UL ) );
   }

   {
      test_ = "Row-major Submatrix::trim( size_t )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 2UL, 1UL, 2UL, 3UL );

      // Increasing the row capacity of the matrix
      sm.reserve( 0UL, 10UL );
      sm.reserve( 1UL, 20UL );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkCapacity( sm  , 30UL );
      checkCapacity( sm  ,  0UL, 10UL );
      checkCapacity( sm  ,  1UL, 20UL );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, 10UL );
      checkCapacity( mat_,  3UL, 20UL );

      // Trimming the 0th row
      sm.trim( 0UL );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkCapacity( sm  , 30UL );
      checkCapacity( sm  ,  0UL, sm.nonZeros( 0UL ) );
      checkCapacity( sm  ,  1UL, 30UL - sm.nonZeros( 0UL ) );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, mat_.nonZeros( 2UL ) );
      checkCapacity( mat_,  3UL, 30UL - mat_.nonZeros( 2UL ) );

      // Trimming the 1st row
      sm.trim( 1UL );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkCapacity( sm  , 30UL );
      checkCapacity( sm  ,  0UL, sm.nonZeros( 0UL ) );
      checkCapacity( sm  ,  1UL, sm.nonZeros( 1UL ) );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, mat_.nonZeros( 2UL ) );
      checkCapacity( mat_,  3UL, mat_.nonZeros( 3UL ) );
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::trim()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 2UL, 3UL, 2UL );

      // Increasing the row capacity of the matrix
      sm.reserve( 0UL, 10UL );
      sm.reserve( 1UL, 20UL );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkCapacity( sm   , 30UL );
      checkCapacity( sm   ,  0UL, 10UL );
      checkCapacity( sm   ,  1UL, 20UL );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, 10UL );
      checkCapacity( tmat_,  3UL, 20UL );

      // Trimming the matrix
      sm.trim();

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkCapacity( sm   , 30UL );
      checkCapacity( sm   ,  0UL, sm.nonZeros( 0UL ) );
      checkCapacity( sm   ,  1UL, sm.nonZeros( 1UL ) );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, tmat_.nonZeros( 2UL ) );
      checkCapacity( tmat_,  3UL, tmat_.nonZeros( 3UL ) );
   }

   {
      test_ = "Column-major Submatrix::trim( size_t )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 2UL, 3UL, 2UL );

      // Increasing the row capacity of the matrix
      sm.reserve( 0UL, 10UL );
      sm.reserve( 1UL, 20UL );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkCapacity( sm   , 30UL );
      checkCapacity( sm   ,  0UL, 10UL );
      checkCapacity( sm   ,  1UL, 20UL );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, 10UL );
      checkCapacity( tmat_,  3UL, 20UL );

      // Trimming the 0th row
      sm.trim( 0UL );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkCapacity( sm   , 30UL );
      checkCapacity( sm   ,  0UL, sm.nonZeros( 0UL ) );
      checkCapacity( sm   ,  1UL, 30UL - sm.nonZeros( 0UL ) );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, tmat_.nonZeros( 2UL ) );
      checkCapacity( tmat_,  3UL, 30UL - tmat_.nonZeros( 2UL ) );

      // Trimming the 1st row
      sm.trim( 1UL );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkCapacity( sm   , 30UL );
      checkCapacity( sm   ,  0UL, sm.nonZeros( 0UL ) );
      checkCapacity( sm   ,  1UL, sm.nonZeros( 1UL ) );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, tmat_.nonZeros( 2UL ) );
      checkCapacity( tmat_,  3UL, tmat_.nonZeros( 3UL ) );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c set() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c set() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testSet()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::set()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 0UL, 1UL, 2UL, 3UL );

      // Setting a non-zero element at the end of the 0th row
      sm.set( 0UL, 2UL, 1 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 11UL );

      if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 1 ||
          sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 0 1 )\n( 1 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the beginning of the 0th row
      sm.set( 0UL, 0UL, 2 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  3UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( sm(0,0) != 2 || sm(0,1) != 0 || sm(0,2) != 1 ||
          sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 0 1 )\n( 1 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the center of the 0th row
      sm.set( 0UL, 1UL, 3 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 13UL );

      if( sm(0,0) != 2 || sm(0,1) != 3 || sm(0,2) != 1 ||
          sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 3 1 )\n( 1 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting an already existing element
      sm.set( 1UL, 0UL, 4 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 13UL );

      if( sm(0,0) != 2 || sm(0,1) != 3 || sm(0,2) != 1 ||
          sm(1,0) != 4 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 3 1 )\n( 4 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::set()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 2UL );

      // Setting a non-zero element at the end of the 0th column
      sm.set( 2UL, 0UL, 1 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  2UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 11UL );

      if( sm(0,0) != 0 || sm(0,1) != 1 ||
          sm(1,0) != 0 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 1 )\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the beginning of the 0th column
      sm.set( 0UL, 0UL, 2 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  3UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 12UL );

      if( sm(0,0) != 2 || sm(0,1) != 1 ||
          sm(1,0) != 0 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 1 )\n( 0 0 )\n( 1 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the center of the 0th column
      sm.set( 1UL, 0UL, 3 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 13UL );

      if( sm(0,0) != 2 || sm(0,1) != 1 ||
          sm(1,0) != 3 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 1 )\n( 3 0 )\n( 1 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting an already existing element
      sm.set( 0UL, 1UL, 4 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 13UL );

      if( sm(0,0) != 2 || sm(0,1) != 4 ||
          sm(1,0) != 3 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 4 )\n( 3 0 )\n( 1 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c insert() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c insert() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testInsert()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::insert()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 0UL, 1UL, 2UL, 3UL );

      // Inserting a non-zero element at the end of the 0th row
      sm.insert( 0UL, 2UL, 1 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  2UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 11UL );

      if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 1 ||
          sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 0 1 )\n( 1 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the beginning of the 0th row
      sm.insert( 0UL, 0UL, 2 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  3UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( sm(0,0) != 2 || sm(0,1) != 0 || sm(0,2) != 1 ||
          sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 0 1 )\n( 1 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the center of the 0th row
      sm.insert( 0UL, 1UL, 3 );

      checkRows    ( sm  ,  2UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  4UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 13UL );

      if( sm(0,0) != 2 || sm(0,1) != 3 || sm(0,2) != 1 ||
          sm(1,0) != 1 || sm(1,1) != 0 || sm(1,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 3 1 )\n( 1 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to insert an already existing element
      try {
         sm.insert( 1UL, 0UL, 4 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting an existing element succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 3 1 )\n( 4 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::insert()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 2UL );

      // Inserting a non-zero element at the end of the 0th column
      sm.insert( 2UL, 0UL, 1 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  2UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 11UL );

      if( sm(0,0) != 0 || sm(0,1) != 1 ||
          sm(1,0) != 0 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 1 )\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the beginning of the 0th column
      sm.insert( 0UL, 0UL, 2 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  3UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 12UL );

      if( sm(0,0) != 2 || sm(0,1) != 1 ||
          sm(1,0) != 0 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 1 )\n( 0 0 )\n( 1 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the center of the 0th column
      sm.insert( 1UL, 0UL, 3 );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  2UL );
      checkNonZeros( sm   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 13UL );

      if( sm(0,0) != 2 || sm(0,1) != 1 ||
          sm(1,0) != 3 || sm(1,1) != 0 ||
          sm(2,0) != 1 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 1 )\n( 3 0 )\n( 1 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to insert an already existing element
      try {
         sm.insert( 0UL, 1UL, 4 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting an existing element succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 2 4 )\n( 3 0 )\n( 1 0 )\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c append() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c append() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testAppend()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::append()";

      // Appending with pre-allocation in each row
      {
         mat_.reset();

         // Initialization check
         SMT sm = blaze::submatrix( mat_, 0UL, 0UL, 4UL, 4UL );
         sm.reserve( 0UL, 2UL );
         sm.reserve( 2UL, 1UL );
         sm.reserve( 3UL, 2UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 0UL );
         checkNonZeros( sm, 0UL, 0UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 0UL );

         // Appending one non-zero element
         sm.append( 2UL, 1UL, 1 );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 1UL );
         checkNonZeros( sm, 0UL, 0UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 1UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(2,1) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 0UL, 0UL, 2 );
         sm.append( 0UL, 3UL, 3 );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 3UL );
         checkNonZeros( sm, 0UL, 2UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 1UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(2,1) != 1 || sm(0,0) != 2 || sm(0,3) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 2 0 0 3 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 3UL, 1UL, 4 );
         sm.append( 3UL, 2UL, 5 );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 5UL );
         checkNonZeros( sm, 0UL, 2UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 1UL );
         checkNonZeros( sm, 3UL, 2UL );

         if( sm(2,1) != 1 || sm(0,0) != 2 || sm(0,3) != 3 ||
             sm(3,1) != 4 || sm(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 2 0 0 3 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 4 5 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Appending with row finalization
      {
         mat_.reset();

         // Initialization check
         SMT sm = blaze::submatrix( mat_, 0UL, 0UL, 4UL, 4UL );
         sm.reserve( 0UL, 2UL );
         sm.reserve( 2UL, 1UL );
         sm.reserve( 3UL, 2UL );

         // Appending one non-zero element
         sm.append( 0UL, 1UL, 1 );
         sm.finalize( 0UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 1UL );
         checkNonZeros( sm, 0UL, 1UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(0,1) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 1UL, 1UL, 2 );
         sm.append( 1UL, 3UL, 3 );
         sm.finalize( 1UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 3UL );
         checkNonZeros( sm, 0UL, 1UL );
         checkNonZeros( sm, 1UL, 2UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(0,1) != 1 || sm(1,1) != 2 || sm(1,3) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n( 0 2 0 3 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 3UL, 0UL, 4 );
         sm.append( 3UL, 1UL, 5 );
         sm.finalize( 1UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 5UL );
         checkNonZeros( sm, 0UL, 1UL );
         checkNonZeros( sm, 1UL, 2UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 2UL );

         if( sm(0,1) != 1 || sm(1,1) != 2 || sm(1,3) != 3 ||
             sm(3,0) != 4 || sm(3,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n( 0 2 0 3 )\n( 0 0 0 0 )\n( 4 5 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::append()";

      // Appending with pre-allocation in each row
      {
         tmat_.reset();

         // Initialization check
         OSMT sm = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 4UL );
         sm.reserve( 0UL, 2UL );
         sm.reserve( 2UL, 1UL );
         sm.reserve( 3UL, 2UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 0UL );
         checkNonZeros( sm, 0UL, 0UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 0UL );

         // Appending one non-zero element
         sm.append( 1UL, 2UL, 1 );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 1UL );
         checkNonZeros( sm, 0UL, 0UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 1UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(1,2) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 1 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 0UL, 0UL, 2 );
         sm.append( 3UL, 0UL, 3 );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 3UL );
         checkNonZeros( sm, 0UL, 2UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 1UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(1,2) != 1 || sm(0,0) != 2 || sm(3,0) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 2 0 0 0 )\n( 0 0 1 0 )\n( 0 0 0 0 )\n( 3 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 1UL, 3UL, 4 );
         sm.append( 2UL, 3UL, 5 );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 5UL );
         checkNonZeros( sm, 0UL, 2UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 1UL );
         checkNonZeros( sm, 3UL, 2UL );

         if( sm(1,2) != 1 || sm(0,0) != 2 || sm(3,0) != 3 ||
             sm(1,3) != 4 || sm(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 2 0 0 0 )\n( 0 0 1 4 )\n( 0 0 0 5 )\n( 3 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Appending with row finalization
      {
         tmat_.reset();

         // Initialization check
         OSMT sm = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 4UL );
         sm.reserve( 0UL, 2UL );
         sm.reserve( 2UL, 1UL );
         sm.reserve( 3UL, 2UL );

         // Appending one non-zero element
         sm.append( 1UL, 0UL, 1 );
         sm.finalize( 0UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 1UL );
         checkNonZeros( sm, 0UL, 1UL );
         checkNonZeros( sm, 1UL, 0UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(1,0) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 1 0 0 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 1UL, 1UL, 2 );
         sm.append( 3UL, 1UL, 3 );
         sm.finalize( 1UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 3UL );
         checkNonZeros( sm, 0UL, 1UL );
         checkNonZeros( sm, 1UL, 2UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 0UL );

         if( sm(1,0) != 1 || sm(1,1) != 2 || sm(3,1) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 1 2 0 0 )\n( 0 0 0 0 )\n( 0 3 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         sm.append( 0UL, 3UL, 4 );
         sm.append( 1UL, 3UL, 5 );
         sm.finalize( 1UL );

         checkRows    ( sm, 4UL );
         checkColumns ( sm, 4UL );
         checkCapacity( sm, 5UL );
         checkNonZeros( sm, 5UL );
         checkNonZeros( sm, 0UL, 1UL );
         checkNonZeros( sm, 1UL, 2UL );
         checkNonZeros( sm, 2UL, 0UL );
         checkNonZeros( sm, 3UL, 2UL );

         if( sm(1,0) != 1 || sm(1,1) != 2 || sm(3,1) != 3 ||
             sm(0,3) != 4 || sm(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 4 )\n( 1 2 0 5 )\n( 0 0 0 0 )\n( 0 3 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c erase() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c erase() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testErase()
{
   //=====================================================================================
   // Row-major index-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::erase( size_t, size_t )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 3UL, 1UL, 2UL, 3UL );

      // Erasing the non-zero element at the end of the 1st row
      sm.erase( 1UL, 2UL );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 5UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( sm(0,0) !=  4 || sm(0,1) != 5 || sm(0,2) != -6 ||
          sm(1,0) != -8 || sm(1,1) != 9 || sm(1,2) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  4  5 -6 )\n( -8  9  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st row
      sm.erase( 1UL, 0UL );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 4UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 8UL );

      if( sm(0,0) != 4 || sm(0,1) != 5 || sm(0,2) != -6 ||
          sm(1,0) != 0 || sm(1,1) != 9 || sm(1,2) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 4  5 -6 )\n( 0  9  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st row
      sm.erase( 1UL, 1UL );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 3UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 7UL );

      if( sm(0,0) != 4 || sm(0,1) != 5 || sm(0,2) != -6 ||
          sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 4  5 -6 )\n( 0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase an already erased element
      sm.erase( 1UL, 2UL );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 3UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 7UL );

      if( sm(0,0) != 4 || sm(0,1) != 5 || sm(0,2) != -6 ||
          sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 4  5 -6 )\n( 0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major iterator-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::erase( size_t, Iterator )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 3UL, 1UL, 2UL, 3UL );

      // Erasing the non-zero element at the end of the 1st row
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.find( 1UL, 2UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 3UL );
         checkNonZeros( sm  , 5UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 9UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) !=  4 || sm(0,1) != 5 || sm(0,2) != -6 ||
             sm(1,0) != -8 || sm(1,1) != 9 || sm(1,2) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4  5 -6 )\n( -8  9  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st row
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.find( 1UL, 0UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 3UL );
         checkNonZeros( sm  , 4UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 8UL );

         if( pos->value() != 9 || pos->index() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 9\n"
                << "   Expected index: 1\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 4 || sm(0,1) != 5 || sm(0,2) != -6 ||
             sm(1,0) != 0 || sm(1,1) != 9 || sm(1,2) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4  5 -6 )\n( 0  9  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st row
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.find( 1UL, 1UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 3UL );
         checkNonZeros( sm  , 3UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 4 || sm(0,1) != 5 || sm(0,2) != -6 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4  5 -6 )\n( 0  0  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an already erased element
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.find( 1UL, 2UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 3UL );
         checkNonZeros( sm  , 3UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 4 || sm(0,1) != 5 || sm(0,2) != -6 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 4  5 -6 )\n( 0  0  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major iterator-range-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::erase( size_t, Iterator, Iterator )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 3UL, 0UL, 2UL, 4UL );

      // Erasing the 0th row
      {
         SMT::Iterator pos = sm.erase( 0UL, sm.begin( 0UL ), sm.end( 0UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 4UL );
         checkNonZeros( sm  , 4UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) !=  0 || sm(0,2) != 0 || sm(0,3) !=  0 ||
             sm(1,0) != 7 || sm(1,1) != -8 || sm(1,2) != 9 || sm(1,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th row failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n( 7 -8  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the first half of the 1st row
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.begin( 1UL ), sm.find( 1UL, 2UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 4UL );
         checkNonZeros( sm  , 2UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 5UL );

         if( pos->value() != 9 || pos->index() != 2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 9\n"
                << "   Expected index: 2\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 0 || sm(0,3) !=  0 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 9 || sm(1,3) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the first half of the 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n( 0  0  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the second half of the 1st row
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.find( 1UL, 2UL ), sm.end( 1UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 4UL );
         checkNonZeros( sm  , 0UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 3UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 0 || sm(0,3) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 0 || sm(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the second half of the 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an empty range
      {
         SMT::Iterator pos = sm.erase( 1UL, sm.begin( 1UL ), sm.begin( 1UL ) );

         checkRows    ( sm  , 2UL );
         checkColumns ( sm  , 4UL );
         checkNonZeros( sm  , 0UL );
         checkRows    ( mat_, 5UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 3UL );

         if( pos != sm.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the given end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) != 0 || sm(0,2) != 0 || sm(0,3) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 || sm(1,2) != 0 || sm(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing an empty range failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major erase function with predicate
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::erase( Predicate )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 3UL, 1UL, 2UL, 3UL );

      // Erasing a selection of elements
      sm.erase( []( int value ) { return value == 4 || value == 10; } );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 4UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != 5 || sm(0,2) != -6 ||
          sm(1,0) != -8 || sm(1,1) != 9 || sm(1,2) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  5 -6 )\n( -8  9  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase all elements with value 1
      sm.erase( []( int value ){ return value == 1; } );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 4UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != 5 || sm(0,2) != -6 ||
          sm(1,0) != -8 || sm(1,1) != 9 || sm(1,2) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing all elements with value 1 failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  5 -6 )\n( -8  9  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major iterator-range-based erase function with predicate
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::erase( size_t, Iterator, Iterator, Predicate )";

      initialize();

      SMT sm = blaze::submatrix( mat_, 3UL, 1UL, 2UL, 3UL );

      // Erasing a selection of elements
      sm.erase( 0UL, sm.begin( 0UL ), sm.find( 0UL, 2UL ),
                []( int value ) { return value == 4 || value == 5; } );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 4UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != 0 || sm(0,2) != -6 ||
          sm(1,0) != -8 || sm(1,1) != 9 || sm(1,2) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  0 -6 )\n( -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase from an empty range
      sm.erase( 1UL, sm.begin( 1UL ), sm.begin( 1UL ), []( int ){ return true; } );

      checkRows    ( sm  , 2UL );
      checkColumns ( sm  , 3UL );
      checkNonZeros( sm  , 4UL );
      checkRows    ( mat_, 5UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != 0 || sm(0,2) != -6 ||
          sm(1,0) != -8 || sm(1,1) != 9 || sm(1,2) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing from an empty range failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  0 -6 )\n( -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major index-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::erase( size_t, size_t )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 3UL, 3UL, 2UL );

      // Erasing the non-zero element at the end of the 1st column
      sm.erase( 2UL, 1UL );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 5UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 9UL );

      if( sm(0,0) !=  4 || sm(0,1) != -8 ||
          sm(1,0) !=  5 || sm(1,1) !=  9 ||
          sm(2,0) != -6 || sm(2,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  4 -8 )\n(  5  9 )\n( -6  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st column
      sm.erase( 0UL, 1UL );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 8UL );

      if( sm(0,0) !=  4 || sm(0,1) != 0 ||
          sm(1,0) !=  5 || sm(1,1) != 9 ||
          sm(2,0) != -6 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  4 0 )\n(  5 9 )\n( -6 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st column
      sm.erase( 1UL, 1UL );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 7UL );

      if( sm(0,0) !=  4 || sm(0,1) != 0 ||
          sm(1,0) !=  5 || sm(1,1) != 0 ||
          sm(2,0) != -6 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  4 0 )\n(  5 0 )\n( -6 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase an already erased element
      sm.erase( 2UL, 1UL );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 7UL );

      if( sm(0,0) !=  4 || sm(0,1) != 0 ||
          sm(1,0) !=  5 || sm(1,1) != 0 ||
          sm(2,0) != -6 || sm(2,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  4 0 )\n(  5 0 )\n( -6 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major iterator-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::erase( size_t, Iterator )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 3UL, 3UL, 2UL );

      // Erasing the non-zero element at the end of the 1st column
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.find( 2UL, 1UL ) );

         checkRows    ( sm   , 3UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 5UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 9UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) !=  4 || sm(0,1) != -8 ||
             sm(1,0) !=  5 || sm(1,1) !=  9 ||
             sm(2,0) != -6 || sm(2,1) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 -8 )\n(  5  9 )\n( -6  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st column
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.find( 0UL, 1UL ) );

         checkRows    ( sm   , 3UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 4UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 8UL );

         if( pos->value() != 9 || pos->index() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 9\n"
                << "   Expected index: 1\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) !=  4 || sm(0,1) != 0 ||
             sm(1,0) !=  5 || sm(1,1) != 9 ||
             sm(2,0) != -6 || sm(2,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 0 )\n(  5 9 )\n( -6 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st column
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.find( 1UL, 1UL ) );

         checkRows    ( sm   , 3UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 3UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 7UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) !=  4 || sm(0,1) != 0 ||
             sm(1,0) !=  5 || sm(1,1) != 0 ||
             sm(2,0) != -6 || sm(2,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 0 )\n(  5 0 )\n( -6 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an already erased element
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.find( 2UL, 1UL ) );

         checkRows    ( sm   , 3UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 3UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 7UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) !=  4 || sm(0,1) != 0 ||
             sm(1,0) !=  5 || sm(1,1) != 0 ||
             sm(2,0) != -6 || sm(2,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  4 0 )\n(  5 0 )\n( -6 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major iterator-range-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::erase( size_t, Iterator, Iterator )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 3UL, 4UL, 2UL );

      // Erasing the 0th column
      {
         OSMT::Iterator pos = sm.erase( 0UL, sm.begin( 0UL ), sm.end( 0UL ) );

         checkRows    ( sm   , 4UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 4UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 7UL );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) !=  7 ||
             sm(1,0) != 0 || sm(1,1) != -8 ||
             sm(2,0) != 0 || sm(2,1) !=  9 ||
             sm(3,0) != 0 || sm(3,1) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0  7 )\n( 0 -8 )\n( 0  9 )\n( 0 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the first half of the 1st column
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.begin( 1UL ), sm.find( 2UL, 1UL ) );

         checkRows    ( sm   , 4UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 5UL );

         if( pos->value() != 9 || pos->index() != 2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 9\n"
                << "   Expected index: 2\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) !=  0 ||
             sm(1,0) != 0 || sm(1,1) !=  0 ||
             sm(2,0) != 0 || sm(2,1) !=  9 ||
             sm(3,0) != 0 || sm(3,1) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0  0 )\n( 0  0 )\n( 0  9 )\n( 0 10 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the second half of the 1st column
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.find( 2UL, 1UL ), sm.end( 1UL ) );

         checkRows    ( sm   , 4UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 3UL );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 ||
             sm(2,0) != 0 || sm(2,1) != 0 ||
             sm(3,0) != 0 || sm(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an empty range
      {
         OSMT::Iterator pos = sm.erase( 1UL, sm.begin( 1UL ), sm.begin( 1UL ) );

         checkRows    ( sm   , 4UL );
         checkColumns ( sm   , 2UL );
         checkNonZeros( sm   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 5UL );
         checkNonZeros( tmat_, 3UL );

         if( pos != sm.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the given end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm(0,0) != 0 || sm(0,1) != 0 ||
             sm(1,0) != 0 || sm(1,1) != 0 ||
             sm(2,0) != 0 || sm(2,1) != 0 ||
             sm(3,0) != 0 || sm(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major erase function with predicate
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::erase( Predicate )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 3UL, 3UL, 2UL );

      // Erasing a selection of values
      sm.erase( []( int value ) { return value == 4 || value == 10; } );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != -8 ||
          sm(1,0) !=  5 || sm(1,1) !=  9 ||
          sm(2,0) != -6 || sm(2,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 -8 )\n(  5  9 )\n( -6  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase all elements with value 1
      sm.erase( []( int value ){ return value == 1; } );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != -8 ||
          sm(1,0) !=  5 || sm(1,1) !=  9 ||
          sm(2,0) != -6 || sm(2,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing all elements with value 1 failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 -8 )\n(  5  9 )\n( -6  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major iterator-range-based erase function with predicate
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::erase( size_t, Iterator, Iterator, Predicate )";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 3UL, 3UL, 2UL );

      // Erasing a selection of elements
      sm.erase( 0UL, sm.begin( 0UL ), sm.find( 2UL, 0UL ),
                []( int value ) { return value == 4 || value == 5; } );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != -8 ||
          sm(1,0) !=  0 || sm(1,1) !=  9 ||
          sm(2,0) != -6 || sm(2,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 -8 )\n(  0  9 )\n( -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase from an empty range
      sm.erase( 1UL, sm.begin( 1UL ), sm.begin( 1UL ), []( int ){ return true; } );

      checkRows    ( sm   , 3UL );
      checkColumns ( sm   , 2UL );
      checkNonZeros( sm   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 5UL );
      checkNonZeros( tmat_, 8UL );

      if( sm(0,0) !=  0 || sm(0,1) != -8 ||
          sm(1,0) !=  0 || sm(1,1) !=  9 ||
          sm(2,0) != -6 || sm(2,1) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing from an empty range failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0 -8 )\n(  0  9 )\n( -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c find() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c find() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testFind()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::find()";

      using ConstIterator = SMT::ConstIterator;

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );

      checkRows    ( sm, 3UL );
      checkColumns ( sm, 2UL );
      checkNonZeros( sm, 4UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 1UL );
      checkNonZeros( sm, 2UL, 2UL );

      // Searching for the first element
      {
         ConstIterator pos( sm.find( 0UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required position = (0,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 0 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for the second element
      {
         ConstIterator pos( sm.find( 1UL, 1UL ) );

         if( pos == sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required position = (1,1)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -3\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for a non-existing non-zero element
      {
         ConstIterator pos( sm.find( 1UL, 0UL ) );

         if( pos != sm.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Non-existing element could be found\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 0\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::find()";

      using ConstIterator = OSMT::ConstIterator;

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );

      checkRows    ( sm, 2UL );
      checkColumns ( sm, 3UL );
      checkNonZeros( sm, 4UL );
      checkNonZeros( sm, 0UL, 1UL );
      checkNonZeros( sm, 1UL, 1UL );
      checkNonZeros( sm, 2UL, 2UL );

      // Searching for the first element
      {
         ConstIterator pos( sm.find( 0UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required position = (0,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 0 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for the second element
      {
         ConstIterator pos( sm.find( 1UL, 2UL ) );

         if( pos == sm.end( 2UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required position = (1,2)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 5\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for a non-existing non-zero element
      {
         ConstIterator pos( sm.find( 1UL, 0UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Non-existing element could be found\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 0\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c lowerBound() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c lowerBound() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testLowerBound()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::lowerBound()";

      using ConstIterator = SMT::ConstIterator;

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 1UL, 4UL );

      checkRows    ( sm, 1UL );
      checkColumns ( sm, 4UL );
      checkNonZeros( sm, 1UL );
      checkNonZeros( sm, 0UL, 1UL );

      // Determining the lower bound for position (0,0)
      {
         ConstIterator pos( sm.lowerBound( 0UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (0,1)
      {
         ConstIterator pos( sm.lowerBound( 0UL, 1UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,1)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (0,2)
      {
         ConstIterator pos( sm.lowerBound( 0UL, 2UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,2)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::lowerBound()";

      using ConstIterator = OSMT::ConstIterator;

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 1UL );

      checkRows    ( sm, 4UL );
      checkColumns ( sm, 1UL );
      checkNonZeros( sm, 1UL );
      checkNonZeros( sm, 0UL, 1UL );

      // Determining the lower bound for position (0,0)
      {
         ConstIterator pos( sm.lowerBound( 0UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (1,0)
      {
         ConstIterator pos( sm.lowerBound( 1UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (1,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (2,0)
      {
         ConstIterator pos( sm.lowerBound( 2UL, 0UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (2,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c upperBound() member function of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c upperBound() member function of the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testUpperBound()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major Submatrix::upperBound()";

      using ConstIterator = SMT::ConstIterator;

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 1UL, 4UL );

      checkRows    ( sm, 1UL );
      checkColumns ( sm, 4UL );
      checkNonZeros( sm, 1UL );
      checkNonZeros( sm, 0UL, 1UL );

      // Determining the upper bound for position (0,0)
      {
         ConstIterator pos( sm.upperBound( 0UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (0,1)
      {
         ConstIterator pos( sm.upperBound( 0UL, 1UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,1)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (0,2)
      {
         ConstIterator pos( sm.upperBound( 0UL, 2UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,2)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major Submatrix::upperBound()";

      using ConstIterator = OSMT::ConstIterator;

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 1UL );

      checkRows    ( sm, 4UL );
      checkColumns ( sm, 1UL );
      checkNonZeros( sm, 1UL );
      checkNonZeros( sm, 0UL, 1UL );

      // Determining the upper bound for position (0,0)
      {
         ConstIterator pos( sm.upperBound( 0UL, 0UL ) );

         if( pos == sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (0,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 1 || pos->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 1\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (1,0)
      {
         ConstIterator pos( sm.upperBound( 1UL, 0UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (1,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (2,0)
      {
         ConstIterator pos( sm.upperBound( 2UL, 0UL ) );

         if( pos != sm.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required position = (2,0)\n"
                << "   Current submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c transpose() member functions of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c transpose() member function of the Submatrix
// specialization. Additionally, it performs a test of self-transpose via the \c trans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testTranspose()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via transpose()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 3UL );

      transpose( sm );

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
          sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 4 ||
          sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -2 0 )\n( 1  0 4 )\n( 0 -3 5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) != -2 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 1 || mat_(2,1) !=  0 || mat_(2,2) != 4 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) != -3 || mat_(3,2) != 5 || mat_(3,3) != -6 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0 -2  0  0 )\n"
                                     "(  1  0  4  0 )\n"
                                     "(  0 -3  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via trans()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 3UL );

      sm = trans( sm );

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
          sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 4 ||
          sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -2 0 )\n( 1  0 4 )\n( 0 -3 5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) != -2 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 1 || mat_(2,1) !=  0 || mat_(2,2) != 4 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) != -3 || mat_(3,2) != 5 || mat_(3,3) != -6 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0 -2  0  0 )\n"
                                     "(  1  0  4  0 )\n"
                                     "(  0 -3  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major self-transpose via transpose()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 3UL, 3UL );

      transpose( sm );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  5UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
          sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -3 ||
          sm(2,0) !=  0 || sm(2,1) != 4 || sm(2,2) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  1  0 )\n( -2  0 -3 )\n(  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 1 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != -2 || tmat_(1,2) != 0 || tmat_(1,3) != -3 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 4 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  1  0  7 )\n"
                                     "(  0 -2  0 -3 -8 )\n"
                                     "(  0  0  4  5  9 )\n"
                                     "(  0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-transpose via trans()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 3UL, 3UL );

      sm = trans( sm );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  5UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
          sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -3 ||
          sm(2,0) !=  0 || sm(2,1) != 4 || sm(2,2) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  1  0 )\n( -2  0 -3 )\n(  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 1 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != -2 || tmat_(1,2) != 0 || tmat_(1,3) != -3 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 4 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  1  0  7 )\n"
                                     "(  0 -2  0 -3 -8 )\n"
                                     "(  0  0  4  5  9 )\n"
                                     "(  0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c ctranspose() member functions of the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c ctranspose() member function of the Submatrix
// specialization. Additionally, it performs a test of self-transpose via the \c ctrans()
// function. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testCTranspose()
{
   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major self-transpose via ctranspose()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 3UL );

      ctranspose( sm );

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
          sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 4 ||
          sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -2 0 )\n( 1  0 4 )\n( 0 -3 5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) != -2 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 1 || mat_(2,1) !=  0 || mat_(2,2) != 4 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) != -3 || mat_(3,2) != 5 || mat_(3,3) != -6 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0 -2  0  0 )\n"
                                     "(  1  0  4  0 )\n"
                                     "(  0 -3  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major self-transpose via ctrans()";

      initialize();

      SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 3UL, 3UL );

      sm = ctrans( sm );

      checkRows    ( sm  ,  3UL );
      checkColumns ( sm  ,  3UL );
      checkNonZeros( sm  ,  5UL );
      checkRows    ( mat_,  5UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( sm(0,0) != 0 || sm(0,1) != -2 || sm(0,2) != 0 ||
          sm(1,0) != 1 || sm(1,1) !=  0 || sm(1,2) != 4 ||
          sm(2,0) != 0 || sm(2,1) != -3 || sm(2,2) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n( 0 -2 0 )\n( 1  0 4 )\n( 0 -3 5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) != -2 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 1 || mat_(2,1) !=  0 || mat_(2,2) != 4 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) != -3 || mat_(3,2) != 5 || mat_(3,3) != -6 ||
          mat_(4,0) != 7 || mat_(4,1) != -8 || mat_(4,2) != 9 || mat_(4,3) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0  0  0  0 )\n"
                                     "(  0 -2  0  0 )\n"
                                     "(  1  0  4  0 )\n"
                                     "(  0 -3  5 -6 )\n"
                                     "(  7 -8  9 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major self-transpose via ctranspose()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 3UL, 3UL );

      ctranspose( sm );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  5UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
          sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -3 ||
          sm(2,0) !=  0 || sm(2,1) != 4 || sm(2,2) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  1  0 )\n( -2  0 -3 )\n(  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 1 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != -2 || tmat_(1,2) != 0 || tmat_(1,3) != -3 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 4 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  1  0  7 )\n"
                                     "(  0 -2  0 -3 -8 )\n"
                                     "(  0  0  4  5  9 )\n"
                                     "(  0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major self-transpose via ctrans()";

      initialize();

      OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 3UL, 3UL );

      sm = ctrans( sm );

      checkRows    ( sm   ,  3UL );
      checkColumns ( sm   ,  3UL );
      checkNonZeros( sm   ,  5UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  5UL );
      checkNonZeros( tmat_, 10UL );

      if( sm(0,0) !=  0 || sm(0,1) != 1 || sm(0,2) !=  0 ||
          sm(1,0) != -2 || sm(1,1) != 0 || sm(1,2) != -3 ||
          sm(2,0) !=  0 || sm(2,1) != 4 || sm(2,2) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n"
             << "   Expected result:\n(  0  1  0 )\n( -2  0 -3 )\n(  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 1 || tmat_(0,3) !=  0 || tmat_(0,4) !=  7 ||
          tmat_(1,0) != 0 || tmat_(1,1) != -2 || tmat_(1,2) != 0 || tmat_(1,3) != -3 || tmat_(1,4) != -8 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 4 || tmat_(2,3) !=  5 || tmat_(2,4) !=  9 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) != -6 || tmat_(3,4) != 10 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Transpose operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0  0  1  0  7 )\n"
                                     "(  0 -2  0 -3 -8 )\n"
                                     "(  0  0  4  5  9 )\n"
                                     "(  0  0  0 -6 10 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testIsDefault()
{
   using blaze::isDefault;


   //=====================================================================================
   // Row-major submatrix tests
   //=====================================================================================

   {
      test_ = "Row-major isDefault() function";

      initialize();

      // isDefault with default submatrix
      {
         SMT sm = blaze::submatrix( mat_, 0UL, 0UL, 1UL, 4UL );

         if( isDefault( sm(0,1) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Submatrix element: " << sm(0,1) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( sm ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default submatrix
      {
         SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 1UL, 4UL );

         if( isDefault( sm ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major submatrix tests
   //=====================================================================================

   {
      test_ = "Column-major isDefault() function";

      initialize();

      // isDefault with default submatrix
      {
         OSMT sm = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 1UL );

         if( isDefault( sm(1,0) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Submatrix element: " << sm(1,0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( sm ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default submatrix
      {
         OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 1UL );

         if( isDefault( sm ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testIsSame()
{
   //=====================================================================================
   // Row-major matrix-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (matrix-based)";

      // isSame with matrix and matching submatrix
      {
         SMT sm = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );

         if( blaze::isSame( sm, mat_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different number of rows)
      {
         SMT sm = blaze::submatrix( mat_, 0UL, 0UL, 4UL, 4UL );

         if( blaze::isSame( sm, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different number of columns)
      {
         SMT sm = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 3UL );

         if( blaze::isSame( sm, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different row index)
      {
         SMT sm = blaze::submatrix( mat_, 1UL, 0UL, 4UL, 4UL );

         if( blaze::isSame( sm, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different column index)
      {
         SMT sm = blaze::submatrix( mat_, 0UL, 1UL, 5UL, 3UL );

         if( blaze::isSame( sm, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         SMT sm1 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );
         SMT sm2 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         SMT sm1 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );
         SMT sm2 = blaze::submatrix( mat_, 0UL, 0UL, 4UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         SMT sm1 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );
         SMT sm2 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         SMT sm1 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );
         SMT sm2 = blaze::submatrix( mat_, 1UL, 0UL, 4UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         SMT sm1 = blaze::submatrix( mat_, 0UL, 0UL, 5UL, 4UL );
         SMT sm2 = blaze::submatrix( mat_, 0UL, 1UL, 5UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major rows-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (rows-based)";

      // isSame with row selection and matching submatrix
      {
         auto rs = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( sm, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different number of rows)
      {
         auto rs = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different number of columns)
      {
         auto rs = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different row index)
      {
         auto rs = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 1UL, 0UL, 2UL, 4UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different column index)
      {
         auto rs = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto rs  = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto rs  = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 0UL, 1UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto rs  = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto rs  = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );
         auto sm2 = blaze::submatrix( rs, 1UL, 0UL, 2UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto rs  = blaze::rows( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 1UL, 2UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major columns-based tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function (columns-based)";

      // isSame with column selection and matching submatrix
      {
         auto cs = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 0UL, 5UL, 3UL );

         if( blaze::isSame( sm, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different number of rows)
      {
         auto cs = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 0UL, 4UL, 3UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different number of columns)
      {
         auto cs = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 0UL, 5UL, 2UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different row index)
      {
         auto cs = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 4UL, 3UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different column index)
      {
         auto cs = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 1UL, 5UL, 2UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto cs  = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto cs  = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto cs  = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto cs  = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 1UL, 0UL, 4UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto cs  = blaze::columns( mat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 1UL, 4UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix-based tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function (matrix-based)";

      // isSame with matrix and matching submatrix
      {
         OSMT sm = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );

         if( blaze::isSame( sm, tmat_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different number of rows)
      {
         OSMT sm = blaze::submatrix( tmat_, 0UL, 0UL, 3UL, 5UL );

         if( blaze::isSame( sm, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different number of columns)
      {
         OSMT sm = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 4UL );

         if( blaze::isSame( sm, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different row index)
      {
         OSMT sm = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 5UL );

         if( blaze::isSame( sm, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching submatrix (different column index)
      {
         OSMT sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 4UL );

         if( blaze::isSame( sm, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         OSMT sm1 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );
         OSMT sm2 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         OSMT sm1 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );
         OSMT sm2 = blaze::submatrix( tmat_, 0UL, 0UL, 3UL, 5UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         OSMT sm1 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );
         OSMT sm2 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         OSMT sm1 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );
         OSMT sm2 = blaze::submatrix( tmat_, 1UL, 0UL, 3UL, 5UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         OSMT sm1 = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 5UL );
         OSMT sm2 = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major rows-based tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function (rows-based)";

      // isSame with row selection and matching submatrix
      {
         auto rs = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 0UL, 3UL, 5UL );

         if( blaze::isSame( sm, rs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different number of rows)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 0UL, 2UL, 5UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different number of columns)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 0UL, 3UL, 4UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different row index)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 1UL, 0UL, 2UL, 5UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with row selection and non-matching submatrix (different column index)
      {
         auto rs = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 4UL );

         if( blaze::isSame( sm, rs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( rs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Row selection:\n" << rs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto rs  = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto rs  = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 0UL, 1UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto rs  = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 3UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto rs  = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );
         auto sm2 = blaze::submatrix( rs, 1UL, 0UL, 2UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto rs  = blaze::rows( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( rs, 0UL, 0UL, 2UL, 4UL );
         auto sm2 = blaze::submatrix( rs, 0UL, 1UL, 2UL, 4UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major columns-based tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function (columns-based)";

      // isSame with column selection and matching submatrix
      {
         auto cs = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 0UL, 4UL, 3UL );

         if( blaze::isSame( sm, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different number of rows)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different number of columns)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 0UL, 4UL, 2UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different row index)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 3UL, 3UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with column selection and non-matching submatrix (different column index)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( cs, 0UL, 1UL, 4UL, 2UL );

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n"
                << "   Submatrix:\n" << sm << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching submatrices
      {
         auto cs  = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of rows)
      {
         auto cs  = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 0UL, 2UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different number of columns)
      {
         auto cs  = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 1UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different row index)
      {
         auto cs  = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 1UL, 0UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching submatrices (different column index)
      {
         auto cs  = blaze::columns( tmat_, { 0UL, 3UL, 2UL } );
         auto sm1 = blaze::submatrix( cs, 0UL, 0UL, 3UL, 2UL );
         auto sm2 = blaze::submatrix( cs, 0UL, 1UL, 3UL, 2UL );

         if( blaze::isSame( sm1, sm2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First submatrix:\n" << sm1 << "\n"
                << "   Second submatrix:\n" << sm2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c blaze::submatrix() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c blaze::submatrix() function with the Submatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testSubmatrix()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major blaze::submatrix() function";

      initialize();

      {
         SMT sm1 = blaze::submatrix( mat_, 1UL, 1UL, 4UL, 3UL );
         SMT sm2 = blaze::submatrix( sm1 , 1UL, 1UL, 3UL, 2UL );

         if( sm2(0,0) != -3 || sm2(0,1) !=  0 ||
             sm2(1,0) !=  5 || sm2(1,1) != -6 ||
             sm2(2,0) !=  9 || sm2(2,1) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm2 << "\n"
                << "   Expected result:\n( -3  0 )\n(  5 -6 )\n(  9 10 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm2.begin(1UL)->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << sm2.begin(1UL)->value() << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         SMT sm1 = blaze::submatrix( mat_, 1UL, 1UL, 4UL, 3UL );
         SMT sm2 = blaze::submatrix( sm1 , 4UL, 1UL, 3UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         SMT sm1 = blaze::submatrix( mat_, 1UL, 1UL, 4UL, 3UL );
         SMT sm2 = blaze::submatrix( sm1 , 1UL, 3UL, 3UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         SMT sm1 = blaze::submatrix( mat_, 1UL, 1UL, 4UL, 3UL );
         SMT sm2 = blaze::submatrix( sm1 , 1UL, 1UL, 4UL, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         SMT sm1 = blaze::submatrix( mat_, 1UL, 1UL, 4UL, 3UL );
         SMT sm2 = blaze::submatrix( sm1 , 1UL, 1UL, 3UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major blaze::submatrix() function";

      initialize();

      {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 3UL, 4UL );
         OSMT sm2 = blaze::submatrix( sm1  , 1UL, 1UL, 2UL, 3UL );

         if( sm2(0,0) != -3 || sm2(0,1) !=  5 || sm2(0,2) !=  9 ||
             sm2(1,0) !=  0 || sm2(1,1) != -6 || sm2(1,2) != 10 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm2 << "\n"
                << "   Expected result:\n( -3  5  9 )\n(  0 -6 10 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm2.begin(1UL)->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << sm2.begin(1UL)->value() << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 3UL, 4UL );
         OSMT sm2 = blaze::submatrix( sm1  , 3UL, 1UL, 2UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 3UL, 4UL );
         OSMT sm2 = blaze::submatrix( sm1  , 1UL, 4UL, 2UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 3UL, 4UL );
         OSMT sm2 = blaze::submatrix( sm1  , 1UL, 1UL, 3UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 3UL, 4UL );
         OSMT sm2 = blaze::submatrix( sm1  , 1UL, 1UL, 2UL, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c row() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testRow()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major row() function";

      initialize();

      {
         SMT  sm1  = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto row1 = blaze::row( sm1, 1UL );

         if( row1[0] != 0 || row1[1] != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 -3 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( row1.begin()->value() != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << row1.begin()->value() << "\n"
                << "   Expected result: -3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         SMT  sm1  = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto row3 = blaze::row( sm1, 3UL );

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
         OSMT sm1  = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto row1 = blaze::row( sm1, 1UL );

         if( row1[0] != 0 || row1[1] != -3 || row1[2] != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( row1.begin()->value() != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << row1.begin()->value() << "\n"
                << "   Expected result: -3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         OSMT sm1  = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto row2 = blaze::row( sm1, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rows() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rows() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testRows()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major rows() function";

      initialize();

      {
         SMT  sm1 = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto rs  = blaze::rows( sm1, { 1UL, 0UL } );

         if( rs(0,0) != 0 || rs(0,1) != -3 ||
             rs(1,0) != 1 || rs(1,1) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 -3 )\n( 1  0 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs.begin( 1UL )->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << rs.begin( 1UL )->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         SMT  sm1 = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto rs  = blaze::rows( sm1, { 3UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
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
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto rs  = blaze::rows( sm1, { 1UL, 0UL } );

         if( rs(0,0) != 0 || rs(0,1) != -3 || rs(0,2) != 5 ||
             rs(1,0) != 1 || rs(1,1) !=  0 || rs(1,2) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 -3  5 )\n( 1  0  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs.begin( 1UL )->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << rs.begin( 1UL )->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto rs  = blaze::rows( sm1, { 2UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c column() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testColumn()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major column() function";

      initialize();

      {
         SMT  sm1  = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto col1 = blaze::column( sm1, 1UL );

         if( col1[0] != 0 || col1[1] != -3 || col1[2] != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 0 -3 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( col1.begin()->value() != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << col1.begin()->value() << "\n"
                << "   Expected result: -3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         SMT  sm1  = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto col2 = blaze::column( sm1, 2UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col2 << "\n";
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
         OSMT sm1  = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto col1 = blaze::column( sm1, 1UL );

         if( col1[0] != 0 || col1[1] != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 0 -3 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( col1.begin()->value() != -3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << col1.begin()->value() << "\n"
                << "   Expected result: -3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         OSMT sm1  = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto col3 = blaze::column( sm1, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c columns() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columns() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testColumns()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major columns() function";

      initialize();

      {
         SMT  sm1 = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto cs  = blaze::columns( sm1, { 1UL, 0UL } );

         if( cs(0,0) !=  0 || cs(0,1) != 1 ||
             cs(1,0) != -3 || cs(1,1) != 0 ||
             cs(2,0) !=  5 || cs(2,1) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n(  0 1 )\n( -3 0 )\n(  5 4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs.begin( 1UL )->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << cs.begin( 1UL )->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         SMT  sm1 = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto cs  = blaze::columns( sm1, { 2UL } );

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
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto cs  = blaze::columns( sm1, { 1UL, 0UL } );

         if( cs(0,0) !=  0 || cs(0,1) != 1 ||
             cs(1,0) != -3 || cs(1,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n(  0 1 )\n( -3 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs.begin( 1UL )->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << cs.begin( 1UL )->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto cs  = blaze::columns( sm1, { 3UL } );

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
/*!\brief Test of the \c band() function with the Submatrix class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c band() function with the Submatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseTest::testBand()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major band() function";

      initialize();

      {
         SMT  sm1 = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto b1  = blaze::band( sm1, -1L );

         if( b1[0] != 0 || b1[1] != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result:\n( 0 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( b1.begin()->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << b1.begin()->value() << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         SMT  sm1 = blaze::submatrix( mat_, 1UL, 1UL, 3UL, 2UL );
         auto b2  = blaze::band( sm1, 2L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b2 << "\n";
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
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto b1  = blaze::band( sm1, 1L );

         if( b1[0] != 0 || b1[1] != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result:\n( 0 5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( b1.begin()->value() != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << b1.begin()->value() << "\n"
                << "   Expected result: 5\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         OSMT sm1 = blaze::submatrix( tmat_, 1UL, 1UL, 2UL, 3UL );
         auto b2  = blaze::band( sm1, -2L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b2 << "\n";
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
void SparseTest::initialize()
{
   // Initializing the row-major compressed matrix
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

   // Initializing the column-major compressed matrix
   tmat_.reset();
   tmat_(1,1) =  1;
   tmat_(0,2) = -2;
   tmat_(2,2) = -3;
   tmat_(1,3) =  4;
   tmat_(2,3) =  5;
   tmat_(3,3) = -6;
   tmat_(0,4) =  7;
   tmat_(1,4) = -8;
   tmat_(2,4) =  9;
   tmat_(3,4) = 10;
}
//*************************************************************************************************

} // namespace submatrix

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
   std::cout << "   Running Submatrix sparse test (part 2)..." << std::endl;

   try
   {
      RUN_SUBMATRIX_SPARSE_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Submatrix sparse test (part 2):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
