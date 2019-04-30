//=================================================================================================
/*!
//  \file src/mathtest/smatsmatmult/AliasingTest.cpp
//  \brief Source file for the sparse matrix/sparse matrix multiplication aliasing test
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
#include <blazetest/mathtest/smatsmatmult/AliasingTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace smatsmatmult {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the aliasing test class.
//
// \exception std::runtime_error Operation error detected.
*/
AliasingTest::AliasingTest()
   : sA3x4_ ( 3UL, 4UL )
   , sB4x3_ ( 4UL, 3UL )
   , sC3x3_ ( 3UL, 3UL )
   , sD3x3_ ( 3UL, 3UL )
   , sE3x3_ ( 3UL, 3UL )
   , tsA3x4_( 3UL, 4UL )
   , tsB4x3_( 4UL, 3UL )
   , tsC3x3_( 3UL, 3UL )
   , tsD3x3_( 3UL, 3UL )
   , tsE3x3_( 3UL, 3UL )
{
   testSMatSMatMult  ();
   testSMatTSMatMult ();
   testTSMatSMatMult ();
   testTSMatTSMatMult();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the sparse matrix/sparse matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the sparse matrix/sparse matrix multiplication.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AliasingTest::testSMatSMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "SMatSMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = sA3x4_ * sB4x3_;
      sA3x4_  = sA3x4_ * sB4x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * sC3x3_;
      sA3x4_  = ( sA3x4_ * sB4x3_ ) * sC3x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * sC3x3_;
      sB4x3_  = ( sA3x4_ * sB4x3_ ) * sC3x3_;

      checkResult( sB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "SMatSMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = sA3x4_ * sB4x3_;
      sB4x3_  = sA3x4_ * sB4x3_;

      checkResult( sB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( sA3x4_ * sB4x3_ );
      sA3x4_  = sC3x3_ * ( sA3x4_ * sB4x3_ );

      checkResult( sA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( sA3x4_ * sB4x3_ );
      sB4x3_  = sC3x3_ * ( sA3x4_ * sB4x3_ );

      checkResult( sB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "SMatSMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ += sC3x3_ * sD3x3_;
      sC3x3_  += sC3x3_ * sD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sC3x3_  += ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sD3x3_  += ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "SMatSMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ += sC3x3_ * sD3x3_;
      sD3x3_  += sC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += sC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  += sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ += sC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  += sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "SMatSMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ -= sC3x3_ * sD3x3_;
      sC3x3_  -= sC3x3_ * sD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sC3x3_  -= ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sD3x3_  -= ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "SMatSMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ -= sC3x3_ * sD3x3_;
      sD3x3_  -= sC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ -= sC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  -= sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ -= sC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  -= sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "SMatSMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ %= sC3x3_ * sD3x3_;
      sC3x3_  %= sC3x3_ * sD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sC3x3_  %= ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sD3x3_  %= ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "SMatSMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ %= sC3x3_ * sD3x3_;
      sD3x3_  %= sC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= sC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  %= sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ %= sC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  %= sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "SMatSMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ *= sC3x3_ * sD3x3_;
      sC3x3_  *= sC3x3_ * sD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sC3x3_  *= ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "SMatSMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * sE3x3_;
      sD3x3_  *= ( sC3x3_ * sD3x3_ ) * sE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "SMatSMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ *= sC3x3_ * sD3x3_;
      sD3x3_  *= sC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ *= sC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  *= sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "SMatSMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ *= sC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  *= sC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the sparse matrix/transpose sparse matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the sparse matrix/transpose sparse matrix
// multiplication. In case an error is detected, a \a std::runtime_error exception is
// thrown.
*/
void AliasingTest::testSMatTSMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "SMatTSMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = sA3x4_ * tsB4x3_;
      sA3x4_  = sA3x4_ * tsB4x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * tsC3x3_;
      sA3x4_  = ( sA3x4_ * sB4x3_ ) * tsC3x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * tsC3x3_;
      sB4x3_  = ( sA3x4_ * sB4x3_ ) * tsC3x3_;

      checkResult( sB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "SMatTSMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = sA3x4_ * tsB4x3_;
      tsB4x3_ = sA3x4_ * tsB4x3_;

      checkResult( tsB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( tsA3x4_ * tsB4x3_ );
      tsA3x4_ = sC3x3_ * ( tsA3x4_ * tsB4x3_ );

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( tsA3x4_ * tsB4x3_ );
      tsB4x3_ = sC3x3_ * ( tsA3x4_ * tsB4x3_ );

      checkResult( tsB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "SMatTSMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ += sC3x3_ * tsD3x3_;
      sC3x3_  += sC3x3_ * tsD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sC3x3_  += ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sD3x3_  += ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "SMatTSMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ += sC3x3_ * tsD3x3_;
      tsD3x3_ += sC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ += sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ += sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tsE3x3_;
      result_ += sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ += sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "SMatTSMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ -= sC3x3_ * tsD3x3_;
      sC3x3_  -= sC3x3_ * tsD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sC3x3_  -= ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sD3x3_  -= ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "SMatTSMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ -= sC3x3_ * tsD3x3_;
      tsD3x3_ -= sC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ -= sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ -= sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tsE3x3_;
      result_ -= sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ -= sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "SMatTSMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ %= sC3x3_ * tsD3x3_;
      sC3x3_  %= sC3x3_ * tsD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sC3x3_  %= ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sD3x3_  %= ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "SMatTSMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ %= sC3x3_ * tsD3x3_;
      tsD3x3_ %= sC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ %= sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ %= sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tsE3x3_;
      result_ %= sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ %= sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "SMatTSMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ *= sC3x3_ * tsD3x3_;
      sC3x3_  *= sC3x3_ * tsD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sC3x3_  *= ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "SMatTSMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * tsE3x3_;
      sD3x3_  *= ( sC3x3_ * sD3x3_ ) * tsE3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "SMatTSMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ *= sC3x3_ * tsD3x3_;
      tsD3x3_ *= sC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ *= sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ *= sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "SMatTSMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tsE3x3_;
      result_ *= sC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ *= sC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the transpose sparse matrix/sparse matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the transpose sparse matrix/sparse matrix
// multiplication. In case an error is detected, a \a std::runtime_error exception is
// thrown.
*/
void AliasingTest::testTSMatSMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "TSMatSMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = tsA3x4_ * sB4x3_;
      tsA3x4_ = tsA3x4_ * sB4x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * sC3x3_;
      tsA3x4_ = ( tsA3x4_ * tsB4x3_ ) * sC3x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * sC3x3_;
      tsB4x3_ = ( tsA3x4_ * tsB4x3_ ) * sC3x3_;

      checkResult( tsB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "TSMatSMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = tsA3x4_ * sB4x3_;
      sB4x3_  = tsA3x4_ * sB4x3_;

      checkResult( sB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( sA3x4_ * sB4x3_ );
      sA3x4_  = tsC3x3_ * ( sA3x4_ * sB4x3_ );

      checkResult( sA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( sA3x4_ * sB4x3_ );
      sB4x3_  = tsC3x3_ * ( sA3x4_ * sB4x3_ );

      checkResult( sB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "TSMatSMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ += tsC3x3_ * sD3x3_;
      tsC3x3_ += tsC3x3_ * sD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsC3x3_ += ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsD3x3_ += ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "TSMatSMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ += tsC3x3_ * sD3x3_;
      sD3x3_  += tsC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  += tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ += tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  += tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "TSMatSMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ -= tsC3x3_ * sD3x3_;
      tsC3x3_ -= tsC3x3_ * sD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsC3x3_ -= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsD3x3_ -= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "TSMatSMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ -= tsC3x3_ * sD3x3_;
      sD3x3_  -= tsC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ -= tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  -= tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ -= tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  -= tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "TSMatSMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ %= tsC3x3_ * sD3x3_;
      tsC3x3_ %= tsC3x3_ * sD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsC3x3_ %= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsD3x3_ %= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "TSMatSMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ %= tsC3x3_ * sD3x3_;
      sD3x3_  %= tsC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  %= tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ %= tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  %= tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "TSMatSMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ *= tsC3x3_ * sD3x3_;
      tsC3x3_ *= tsC3x3_ * sD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsC3x3_ *= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "TSMatSMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;
      tsD3x3_ *= ( tsC3x3_ * tsD3x3_ ) * sE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "TSMatSMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  sD3x3_;
      result_ *= tsC3x3_ * sD3x3_;
      sD3x3_  *= tsC3x3_ * sD3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ *= tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sD3x3_  *= tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "TSMatSMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ *= tsC3x3_ * ( sD3x3_ * sE3x3_ );
      sE3x3_  *= tsC3x3_ * ( sD3x3_ * sE3x3_ );

      checkResult( sE3x3_, result_ );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the transpose sparse matrix/transpose sparse matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the transpose sparse matrix/transpose sparse
// matrix multiplication. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void AliasingTest::testTSMatTSMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "TSMatTSMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = tsA3x4_ * tsB4x3_;
      tsA3x4_ = tsA3x4_ * tsB4x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * tsC3x3_;
      tsA3x4_ = ( tsA3x4_ * tsB4x3_ ) * tsC3x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * tsC3x3_;
      tsB4x3_ = ( tsA3x4_ * tsB4x3_ ) * tsC3x3_;

      checkResult( tsB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "TSMatTSMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = tsA3x4_ * tsB4x3_;
      tsB4x3_ = tsA3x4_ * tsB4x3_;

      checkResult( tsB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( tsA3x4_ * tsB4x3_ );
      tsA3x4_ = tsC3x3_ * ( tsA3x4_ * tsB4x3_ );

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( tsA3x4_ * tsB4x3_ );
      tsB4x3_ = tsC3x3_ * ( tsA3x4_ * tsB4x3_ );

      checkResult( tsB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "TSMatTSMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ += tsC3x3_ * tsD3x3_;
      tsC3x3_ += tsC3x3_ * tsD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsC3x3_ += ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsD3x3_ += ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "TSMatTSMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ += tsC3x3_ * tsD3x3_;
      tsD3x3_ += tsC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ += tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ += tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ += tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "TSMatTSMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ -= tsC3x3_ * tsD3x3_;
      tsC3x3_ -= tsC3x3_ * tsD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsC3x3_ -= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsD3x3_ -= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "TSMatTSMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ -= tsC3x3_ * tsD3x3_;
      tsD3x3_ -= tsC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ -= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ -= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tsE3x3_;
      result_ -= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ -= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "TSMatTSMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ %= tsC3x3_ * tsD3x3_;
      tsC3x3_ %= tsC3x3_ * tsD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsC3x3_ %= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsD3x3_ %= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "TSMatTSMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ %= tsC3x3_ * tsD3x3_;
      tsD3x3_ %= tsC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ %= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  sE3x3_;
      result_ %= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ %= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "TSMatTSMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ *= tsC3x3_ * tsD3x3_;
      tsC3x3_ *= tsC3x3_ * tsD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsC3x3_ *= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTSMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;
      tsD3x3_ *= ( tsC3x3_ * tsD3x3_ ) * tsE3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "TSMatTSMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  tsD3x3_;
      result_ *= tsC3x3_ * tsD3x3_;
      tsD3x3_ *= tsC3x3_ * tsD3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ *= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsD3x3_ *= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTSMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tsE3x3_;
      result_ *= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );
      tsE3x3_ *= tsC3x3_ * ( tsD3x3_ * tsE3x3_ );

      checkResult( tsE3x3_, result_ );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Initialization of all member vectors and matrices.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function initializes all member vectors and matrices to specific predetermined values.
*/
void AliasingTest::initialize()
{
   // Initializing the first row-major sparse matrix
   sA3x4_.resize( 3UL, 4UL, false );
   sA3x4_.reset();
   sA3x4_(0,0) = -1;
   sA3x4_(0,2) = -2;
   sA3x4_(1,1) =  2;
   sA3x4_(1,2) = -3;
   sA3x4_(1,3) =  1;
   sA3x4_(2,1) =  1;
   sA3x4_(2,2) =  2;
   sA3x4_(2,3) =  2;

   // Initializing the second row-major sparse matrix
   sB4x3_.resize( 4UL, 3UL, false );
   sB4x3_.reset();
   sB4x3_(0,0) =  1;
   sB4x3_(0,2) = -3;
   sB4x3_(1,1) = -1;
   sB4x3_(2,1) =  2;
   sB4x3_(2,2) =  1;
   sB4x3_(3,0) =  2;
   sB4x3_(3,1) =  1;
   sB4x3_(3,2) = -2;

   // Initializing the third row-major sparse matrix
   sC3x3_.resize( 3UL, 3UL, false );
   sC3x3_.reset();
   sC3x3_(0,0) =  1;
   sC3x3_(0,2) =  2;
   sC3x3_(1,1) =  3;
   sC3x3_(1,2) = -1;
   sC3x3_(2,0) = -1;
   sC3x3_(2,2) =  2;

   // Initializing the fourth row-major sparse matrix
   sD3x3_.resize( 3UL, 3UL, false );
   sD3x3_.reset();
   sD3x3_(0,1) = -1;
   sD3x3_(1,0) =  1;
   sD3x3_(1,1) = -2;
   sD3x3_(1,2) =  2;
   sD3x3_(2,2) = -3;

   // Initializing the fifth row-major sparse matrix
   sE3x3_.resize( 3UL, 3UL, false );
   sE3x3_.reset();
   sE3x3_(0,0) =  2;
   sE3x3_(1,1) =  1;
   sE3x3_(1,2) = -2;
   sE3x3_(2,0) =  1;

   // Initializing the first column-major sparse matrix
   tsA3x4_.resize( 3UL, 4UL, false );
   tsA3x4_.reset();
   tsA3x4_(0,0) = -1;
   tsA3x4_(0,2) = -2;
   tsA3x4_(1,1) =  2;
   tsA3x4_(1,2) = -3;
   tsA3x4_(1,3) =  1;
   tsA3x4_(2,1) =  1;
   tsA3x4_(2,2) =  2;
   tsA3x4_(2,3) =  2;

   // Initializing the second column-major sparse matrix
   tsB4x3_.resize( 4UL, 3UL, false );
   tsB4x3_.reset();
   tsB4x3_(0,0) =  1;
   tsB4x3_(0,2) = -3;
   tsB4x3_(1,1) = -1;
   tsB4x3_(2,1) =  2;
   tsB4x3_(2,2) =  1;
   tsB4x3_(3,0) =  2;
   tsB4x3_(3,1) =  1;
   tsB4x3_(3,2) = -2;

   // Initializing the third column-major sparse matrix
   tsC3x3_.resize( 3UL, 3UL, false );
   tsC3x3_.reset();
   tsC3x3_(0,0) =  1;
   tsC3x3_(0,2) =  2;
   tsC3x3_(1,1) =  3;
   tsC3x3_(1,2) = -1;
   tsC3x3_(2,0) = -1;
   tsC3x3_(2,2) =  2;

   // Initializing the fourth column-major sparse matrix
   tsD3x3_.resize( 3UL, 3UL, false );
   tsD3x3_.reset();
   tsD3x3_(0,1) = -1;
   tsD3x3_(1,0) =  1;
   tsD3x3_(1,1) = -2;
   tsD3x3_(1,2) =  2;
   tsD3x3_(2,2) = -3;

   // Initializing the fifth column-major sparse matrix
   tsE3x3_.resize( 3UL, 3UL, false );
   tsE3x3_.reset();
   tsE3x3_(0,0) =  2;
   tsE3x3_(1,1) =  1;
   tsE3x3_(1,2) = -2;
   tsE3x3_(2,0) =  1;
}
//*************************************************************************************************

} // namespace smatsmatmult

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
   std::cout << "   Running aliasing test..." << std::endl;

   try
   {
      RUN_SMATSMATMULT_ALIASING_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during aliasing test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
