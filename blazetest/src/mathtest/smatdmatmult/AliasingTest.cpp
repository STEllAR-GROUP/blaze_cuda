//=================================================================================================
/*!
//  \file src/mathtest/smatdmatmult/AliasingTest.cpp
//  \brief Source file for the sparse matrix/dense matrix multiplication aliasing test
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
#include <blazetest/mathtest/smatdmatmult/AliasingTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace smatdmatmult {

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
   , tsA3x4_( 3UL, 4UL )
   , tsB4x3_( 4UL, 3UL )
   , tsC3x3_( 3UL, 3UL )
   , tsD3x3_( 3UL, 3UL )
   , dA3x4_ ( 3UL, 4UL )
   , dB4x3_ ( 4UL, 3UL )
   , dC3x3_ ( 3UL, 3UL )
   , dD3x3_ ( 3UL, 3UL )
   , tdA3x4_( 3UL, 4UL )
   , tdB4x3_( 4UL, 3UL )
   , tdC3x3_( 3UL, 3UL )
   , tdD3x3_( 3UL, 3UL )
{
   testSMatDMatMult  ();
   testSMatTDMatMult ();
   testTSMatDMatMult ();
   testTSMatTDMatMult();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the sparse matrix/dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the sparse matrix/dense matrix multiplication.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void AliasingTest::testSMatDMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "SMatDMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = sA3x4_ * dB4x3_;
      sA3x4_  = sA3x4_ * dB4x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * dC3x3_;
      sA3x4_  = ( sA3x4_ * sB4x3_ ) * dC3x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * dC3x3_;
      sB4x3_  = ( sA3x4_ * sB4x3_ ) * dC3x3_;

      checkResult( sB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "SMatDMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = sA3x4_ * dB4x3_;
      dB4x3_  = sA3x4_ * dB4x3_;

      checkResult( dB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( dA3x4_ * dB4x3_ );
      dA3x4_  = sC3x3_ * ( dA3x4_ * dB4x3_ );

      checkResult( dA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( dA3x4_ * dB4x3_ );
      dB4x3_  = sC3x3_ * ( dA3x4_ * dB4x3_ );

      checkResult( dB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "SMatDMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ += sC3x3_ * dD3x3_;
      sC3x3_  += sC3x3_ * dD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sC3x3_  += ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sD3x3_  += ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "SMatDMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ += sC3x3_ * dD3x3_;
      dD3x3_  += sC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ += sC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  += sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ += sC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  += sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "SMatDMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ -= sC3x3_ * dD3x3_;
      sC3x3_  -= sC3x3_ * dD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sC3x3_  -= ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sD3x3_  -= ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "SMatDMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ -= sC3x3_ * dD3x3_;
      dD3x3_  -= sC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ -= sC3x3_ * ( dC3x3_ * dD3x3_ );
      dC3x3_  -= sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dC3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ -= sC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  -= sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "SMatDMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ %= sC3x3_ * dD3x3_;
      sC3x3_  %= sC3x3_ * dD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sC3x3_  %= ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sD3x3_  %= ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "SMatDMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ %= sC3x3_ * dD3x3_;
      dD3x3_  %= sC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ %= sC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  %= sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ %= sC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  %= sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "SMatDMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ *= sC3x3_ * dD3x3_;
      sC3x3_  *= sC3x3_ * dD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sC3x3_  *= ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "SMatDMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * dC3x3_;
      sD3x3_  *= ( sC3x3_ * sD3x3_ ) * dC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "SMatDMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ *= sC3x3_ * dD3x3_;
      dD3x3_  *= sC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ *= sC3x3_ * ( dC3x3_ * dD3x3_ );
      dC3x3_  *= sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dC3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "SMatDMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ *= sC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  *= sC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the sparse matrix/transpose dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the sparse matrix/transpose dense matrix
// multiplication. In case an error is detected, a \a std::runtime_error exception is
// thrown.
*/
void AliasingTest::testSMatTDMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "SMatTDMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = sA3x4_ * tdB4x3_;
      sA3x4_  = sA3x4_ * tdB4x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * tdC3x3_;
      sA3x4_  = ( sA3x4_ * sB4x3_ ) * tdC3x3_;

      checkResult( sA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( sA3x4_ * sB4x3_ ) * tdC3x3_;
      sB4x3_  = ( sA3x4_ * sB4x3_ ) * tdC3x3_;

      checkResult( sB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "SMatTDMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = sA3x4_ * tdB4x3_;
      tdB4x3_ = sA3x4_ * tdB4x3_;

      checkResult( tdB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( tdA3x4_ * tdB4x3_ );
      tdA3x4_ = sC3x3_ * ( tdA3x4_ * tdB4x3_ );

      checkResult( tdA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = sC3x3_ * ( tdA3x4_ * tdB4x3_ );
      tdB4x3_ = sC3x3_ * ( tdA3x4_ * tdB4x3_ );

      checkResult( tdB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "SMatTDMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ += sC3x3_ * tdD3x3_;
      sC3x3_  += sC3x3_ * tdD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sC3x3_  += ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ += ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sD3x3_  += ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "SMatTDMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ += sC3x3_ * tdD3x3_;
      tdD3x3_ += sC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tdC3x3_;
      result_ += sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ += sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tdD3x3_;
      result_ += sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ += sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "SMatTDMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ -= sC3x3_ * tdD3x3_;
      sC3x3_  -= sC3x3_ * tdD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sC3x3_  -= ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ -= ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sD3x3_  -= ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "SMatTDMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ -= sC3x3_ * tdD3x3_;
      tdD3x3_ -= sC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tdC3x3_;
      result_ -= sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ -= sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tdD3x3_;
      result_ -= sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ -= sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "SMatTDMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ %= sC3x3_ * tdD3x3_;
      sC3x3_  %= sC3x3_ * tdD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sC3x3_  %= ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ %= ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sD3x3_  %= ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "SMatTDMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ %= sC3x3_ * tdD3x3_;
      tdD3x3_ %= sC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tdC3x3_;
      result_ %= sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ %= sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tdD3x3_;
      result_ %= sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ %= sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "SMatTDMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  sC3x3_;
      result_ *= sC3x3_ * tdD3x3_;
      sC3x3_  *= sC3x3_ * tdD3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  sC3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sC3x3_  *= ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "SMatTDMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  sD3x3_;
      result_ *= ( sC3x3_ * sD3x3_ ) * tdC3x3_;
      sD3x3_  *= ( sC3x3_ * sD3x3_ ) * tdC3x3_;

      checkResult( sD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "SMatTDMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ *= sC3x3_ * tdD3x3_;
      tdD3x3_ *= sC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tdC3x3_;
      result_ *= sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ *= sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "SMatTDMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tdD3x3_;
      result_ *= sC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ *= sC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the transpose sparse matrix/dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the transpose sparse matrix/dense matrix
// multiplication. In case an error is detected, a \a std::runtime_error exception is
// thrown.
*/
void AliasingTest::testTSMatDMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "TSMatDMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = tsA3x4_ * dB4x3_;
      tsA3x4_ = tsA3x4_ * dB4x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * dC3x3_;
      tsA3x4_ = ( tsA3x4_ * tsB4x3_ ) * dC3x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * dC3x3_;
      tsB4x3_ = ( tsA3x4_ * tsB4x3_ ) * dC3x3_;

      checkResult( tsB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "TSMatDMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = tsA3x4_ * dB4x3_;
      dB4x3_  = tsA3x4_ * dB4x3_;

      checkResult( dB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( dA3x4_ * dB4x3_ );
      dA3x4_  = tsC3x3_ * ( dA3x4_ * dB4x3_ );

      checkResult( dA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( dA3x4_ * dB4x3_ );
      dB4x3_  = tsC3x3_ * ( dA3x4_ * dB4x3_ );

      checkResult( dB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "TSMatDMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ += tsC3x3_ * dD3x3_;
      tsC3x3_ += tsC3x3_ * dD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsC3x3_ += ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsD3x3_ += ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "TSMatDMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ += tsC3x3_ * dD3x3_;
      dD3x3_  += tsC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ += tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dC3x3_  += tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dC3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ += tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  += tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "TSMatDMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ -= tsC3x3_ * dD3x3_;
      tsC3x3_ -= tsC3x3_ * dD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsC3x3_ -= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsD3x3_ -= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "TSMatDMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ -= tsC3x3_ * dD3x3_;
      dD3x3_  -= tsC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ -= tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dC3x3_  -= tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dC3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ -= tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  -= tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "TSMatDMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ %= tsC3x3_ * dD3x3_;
      tsC3x3_ %= tsC3x3_ * dD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsC3x3_ %= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsD3x3_ %= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "TSMatDMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ %= tsC3x3_ * dD3x3_;
      dD3x3_  %= tsC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ %= tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dC3x3_  %= tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dC3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ %= tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  %= tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "TSMatDMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ *= tsC3x3_ * dD3x3_;
      tsC3x3_ *= tsC3x3_ * dD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsC3x3_ *= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "TSMatDMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;
      tsD3x3_ *= ( tsC3x3_ * tsD3x3_ ) * dC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "TSMatDMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  dD3x3_;
      result_ *= tsC3x3_ * dD3x3_;
      dD3x3_  *= tsC3x3_ * dD3x3_;

      checkResult( dD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ *= tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dC3x3_  *= tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dC3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "TSMatDMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ *= tsC3x3_ * ( dC3x3_ * dD3x3_ );
      dD3x3_  *= tsC3x3_ * ( dC3x3_ * dD3x3_ );

      checkResult( dD3x3_, result_ );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the transpose sparse matrix/transpose dense matrix multiplication.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs aliasing tests for the transpose sparse matrix/transpose dense
// matrix multiplication. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void AliasingTest::testTSMatTDMatMult()
{
   //=====================================================================================
   // Multiplication
   //=====================================================================================

   // Assignment to left-hand side operand
   {
      test_ = "TSMatTDMatMult - Assignment to left-hand side operand";

      initialize();

      result_ = tsA3x4_ * tdB4x3_;
      tsA3x4_ = tsA3x4_ * tdB4x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Assignment to first operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * tdC3x3_;
      tsA3x4_ = ( tsA3x4_ * tsB4x3_ ) * tdC3x3_;

      checkResult( tsA3x4_, result_ );
   }

   // Assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Assignment to second operand of left-hand side compound";

      initialize();

      result_ = ( tsA3x4_ * tsB4x3_ ) * tdC3x3_;
      tsB4x3_ = ( tsA3x4_ * tsB4x3_ ) * tdC3x3_;

      checkResult( tsB4x3_, result_ );
   }

   // Assignment to right-hand side operand
   {
      test_ = "TSMatTDMatMult - Assignment to right-hand side operand";

      initialize();

      result_ = tsA3x4_ * tdB4x3_;
      tdB4x3_ = tsA3x4_ * tdB4x3_;

      checkResult( tdB4x3_, result_ );
   }

   // Assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Assignment to first operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( tdA3x4_ * tdB4x3_ );
      tdA3x4_ = tsC3x3_ * ( tdA3x4_ * tdB4x3_ );

      checkResult( tdA3x4_, result_ );
   }

   // Assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Assignment to second operand of right-hand side compound";

      initialize();

      result_ = tsC3x3_ * ( tdA3x4_ * tdB4x3_ );
      tdB4x3_ = tsC3x3_ * ( tdA3x4_ * tdB4x3_ );

      checkResult( tdB4x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with addition assignment
   //=====================================================================================

   // Addition assignment to left-hand side operand
   {
      test_ = "TSMatTDMatMult - Addition assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ += tsC3x3_ * tdD3x3_;
      tsC3x3_ += tsC3x3_ * tdD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Addition assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsC3x3_ += ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Addition assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Addition assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ += ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsD3x3_ += ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Addition assignment to right-hand side operand
   {
      test_ = "TSMatTDMatMult - Addition assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ += tsC3x3_ * tdD3x3_;
      tdD3x3_ += tsC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Addition assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Addition assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ += tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ += tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Addition assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Addition assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ += tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ += tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with subtraction assignment
   //=====================================================================================

   // Subtraction assignment to left-hand side operand
   {
      test_ = "TSMatTDMatMult - Subtraction assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ -= tsC3x3_ * tdD3x3_;
      tsC3x3_ -= tsC3x3_ * tdD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Subtraction assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsC3x3_ -= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Subtraction assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Subtraction assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ -= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsD3x3_ -= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Subtraction assignment to right-hand side operand
   {
      test_ = "TSMatTDMatMult - Subtraction assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ -= tsC3x3_ * tdD3x3_;
      tdD3x3_ -= tsC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Subtraction assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Subtraction assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tdC3x3_;
      result_ -= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ -= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Subtraction assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Subtraction assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tdD3x3_;
      result_ -= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ -= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with Schur product assignment
   //=====================================================================================

   // Schur product assignment to left-hand side operand
   {
      test_ = "TSMatTDMatMult - Schur product assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ %= tsC3x3_ * tdD3x3_;
      tsC3x3_ %= tsC3x3_ * tdD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Schur product assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsC3x3_ %= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Schur product assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Schur product assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ %= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsD3x3_ %= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Schur product assignment to right-hand side operand
   {
      test_ = "TSMatTDMatMult - Schur product assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ %= tsC3x3_ * tdD3x3_;
      tdD3x3_ %= tsC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Schur product assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Schur product assignment to first operand of right-hand side compound";

      initialize();

      result_ =  dC3x3_;
      result_ %= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ %= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Schur product assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Schur product assignment to second operand of right-hand side compound";

      initialize();

      result_ =  dD3x3_;
      result_ %= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ %= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
   }


   //=====================================================================================
   // Multiplication with multiplication assignment
   //=====================================================================================

   // Multiplication assignment to left-hand side operand
   {
      test_ = "TSMatTDMatMult - Multiplication assignment to left-hand side operand";

      initialize();

      result_ =  tsC3x3_;
      result_ *= tsC3x3_ * tdD3x3_;
      tsC3x3_ *= tsC3x3_ * tdD3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to first operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Multiplication assignment to first operand of left-hand side compound";

      initialize();

      result_ =  tsC3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsC3x3_ *= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsC3x3_, result_ );
   }

   // Multiplication assignment to second operand of left-hand side compound
   {
      test_ = "TSMatTDMatMult - Multiplication assignment to second operand of left-hand side compound";

      initialize();

      result_ =  tsD3x3_;
      result_ *= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;
      tsD3x3_ *= ( tsC3x3_ * tsD3x3_ ) * tdC3x3_;

      checkResult( tsD3x3_, result_ );
   }

   // Multiplication assignment to right-hand side operand
   {
      test_ = "TSMatTDMatMult - Multiplication assignment to right-hand side operand";

      initialize();

      result_ =  tdD3x3_;
      result_ *= tsC3x3_ * tdD3x3_;
      tdD3x3_ *= tsC3x3_ * tdD3x3_;

      checkResult( tdD3x3_, result_ );
   }

   // Multiplication assignment to first operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Multiplication assignment to first operand of right-hand side compound";

      initialize();

      result_ =  tdC3x3_;
      result_ *= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdC3x3_ *= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdC3x3_, result_ );
   }

   // Multiplication assignment to second operand of right-hand side compound
   {
      test_ = "TSMatTDMatMult - Multiplication assignment to second operand of right-hand side compound";

      initialize();

      result_ =  tdD3x3_;
      result_ *= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );
      tdD3x3_ *= tsC3x3_ * ( tdC3x3_ * tdD3x3_ );

      checkResult( tdD3x3_, result_ );
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
   //=====================================================================================
   // Initialization of the sparse matrices
   //=====================================================================================

   // Initializing the first row-major dense matrix
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

   // Initializing the second row-major dense matrix
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

   // Initializing the third row-major dense matrix
   sC3x3_.resize( 3UL, 3UL, false );
   sC3x3_.reset();
   sC3x3_(0,0) =  1;
   sC3x3_(0,2) =  2;
   sC3x3_(1,1) =  3;
   sC3x3_(1,2) = -1;
   sC3x3_(2,0) = -1;
   sC3x3_(2,2) =  2;

   // Initializing the fourth row-major dense matrix
   sD3x3_.resize( 3UL, 3UL, false );
   sD3x3_.reset();
   sD3x3_(0,1) = -1;
   sD3x3_(1,0) =  1;
   sD3x3_(1,1) = -2;
   sD3x3_(1,2) =  2;
   sD3x3_(2,2) = -3;

   // Initializing the first column-major dense matrix
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

   // Initializing the second column-major dense matrix
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

   // Initializing the third column-major dense matrix
   tsC3x3_.resize( 3UL, 3UL, false );
   tsC3x3_.reset();
   tsC3x3_(0,0) =  1;
   tsC3x3_(0,2) =  2;
   tsC3x3_(1,1) =  3;
   tsC3x3_(1,2) = -1;
   tsC3x3_(2,0) = -1;
   tsC3x3_(2,2) =  2;

   // Initializing the fourth column-major dense matrix
   tsD3x3_.resize( 3UL, 3UL, false );
   tsD3x3_.reset();
   tsD3x3_(0,1) = -1;
   tsD3x3_(1,0) =  1;
   tsD3x3_(1,1) = -2;
   tsD3x3_(1,2) =  2;
   tsD3x3_(2,2) = -3;


   //=====================================================================================
   // Initialization of the dense matrices
   //=====================================================================================

   // Initializing the first row-major dense matrix
   dA3x4_.resize( 3UL, 4UL, false );
   dA3x4_(0,0) = -1;
   dA3x4_(0,1) =  0;
   dA3x4_(0,2) = -2;
   dA3x4_(0,3) =  0;
   dA3x4_(1,0) =  0;
   dA3x4_(1,1) =  2;
   dA3x4_(1,2) = -3;
   dA3x4_(1,3) =  1;
   dA3x4_(2,0) =  0;
   dA3x4_(2,1) =  1;
   dA3x4_(2,2) =  2;
   dA3x4_(2,3) =  2;

   // Initializing the second row-major dense matrix
   dB4x3_.resize( 4UL, 3UL, false );
   dB4x3_(0,0) =  1;
   dB4x3_(0,1) =  0;
   dB4x3_(0,2) = -3;
   dB4x3_(1,0) =  0;
   dB4x3_(1,1) = -1;
   dB4x3_(1,2) =  0;
   dB4x3_(2,0) =  0;
   dB4x3_(2,1) =  2;
   dB4x3_(2,2) =  1;
   dB4x3_(3,0) =  2;
   dB4x3_(3,1) =  1;
   dB4x3_(3,2) = -2;

   // Initializing the third row-major dense matrix
   dC3x3_.resize( 3UL, 3UL, false );
   dC3x3_(0,0) =  1;
   dC3x3_(0,1) =  0;
   dC3x3_(0,2) =  2;
   dC3x3_(1,0) =  0;
   dC3x3_(1,1) =  3;
   dC3x3_(1,2) = -1;
   dC3x3_(2,0) = -1;
   dC3x3_(2,1) =  0;
   dC3x3_(2,2) =  2;

   // Initializing the fourth row-major dense matrix
   dD3x3_.resize( 3UL, 3UL, false );
   dD3x3_(0,0) =  0;
   dD3x3_(0,1) = -1;
   dD3x3_(0,2) =  0;
   dD3x3_(1,0) =  1;
   dD3x3_(1,1) = -2;
   dD3x3_(1,2) =  2;
   dD3x3_(2,0) =  0;
   dD3x3_(2,1) =  0;
   dD3x3_(2,2) = -3;

   // Initializing the first column-major dense matrix
   tdA3x4_.resize( 3UL, 4UL, false );
   tdA3x4_.resize( 3UL, 4UL, false );
   tdA3x4_(0,0) = -1;
   tdA3x4_(0,1) =  0;
   tdA3x4_(0,2) = -2;
   tdA3x4_(0,3) =  0;
   tdA3x4_(1,0) =  0;
   tdA3x4_(1,1) =  2;
   tdA3x4_(1,2) = -3;
   tdA3x4_(1,3) =  1;
   tdA3x4_(2,0) =  0;
   tdA3x4_(2,1) =  1;
   tdA3x4_(2,2) =  2;
   tdA3x4_(2,3) =  2;

   // Initializing the second column-major dense matrix
   tdB4x3_.resize( 4UL, 3UL, false );
   tdB4x3_(0,0) =  1;
   tdB4x3_(0,1) =  0;
   tdB4x3_(0,2) = -3;
   tdB4x3_(1,0) =  0;
   tdB4x3_(1,1) = -1;
   tdB4x3_(1,2) =  0;
   tdB4x3_(2,0) =  0;
   tdB4x3_(2,1) =  2;
   tdB4x3_(2,2) =  1;
   tdB4x3_(3,0) =  2;
   tdB4x3_(3,1) =  1;
   tdB4x3_(3,2) = -2;

   // Initializing the third column-major dense matrix
   tdC3x3_.resize( 3UL, 3UL, false );
   tdC3x3_(0,0) =  1;
   tdC3x3_(0,1) =  0;
   tdC3x3_(0,2) =  2;
   tdC3x3_(1,0) =  0;
   tdC3x3_(1,1) =  3;
   tdC3x3_(1,2) = -1;
   tdC3x3_(2,0) = -1;
   tdC3x3_(2,1) =  0;
   tdC3x3_(2,2) =  2;

   // Initializing the fourth column-major dense matrix
   tdD3x3_.resize( 3UL, 3UL, false );
   tdD3x3_(0,0) =  0;
   tdD3x3_(0,1) = -1;
   tdD3x3_(0,2) =  0;
   tdD3x3_(1,0) =  1;
   tdD3x3_(1,1) = -2;
   tdD3x3_(1,2) =  2;
   tdD3x3_(2,0) =  0;
   tdD3x3_(2,1) =  0;
   tdD3x3_(2,2) = -3;
}
//*************************************************************************************************

} // namespace smatdmatmult

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
      RUN_SMATDMATMULT_ALIASING_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during aliasing test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
