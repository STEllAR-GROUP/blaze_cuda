//=================================================================================================
/*!
//  \file src/mathtest/rows/SparseSymmetricTest.cpp
//  \brief Source file for the Rows sparse symmetric test
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
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Views.h>
#include <blazetest/mathtest/rows/SparseSymmetricTest.h>

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
/*!\brief Constructor for the Rows sparse symmetric test.
//
// \exception std::runtime_error Operation error detected.
*/
SparseSymmetricTest::SparseSymmetricTest()
   : mat_ ( 4UL )
   , tmat_( 4UL )
{
   testConstructors();
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
   testIsDefault();
   testIsSame();
   testSubmatrix();
   testRow();
   testRows();
   testColumn();
   testBand();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the Rows constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the Rows specialization. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testConstructors()
{
   using blaze::index_sequence;
   using blaze::initializer_list;


   //=====================================================================================
   // Row-major setup via index_sequence
   //=====================================================================================

   {
      test_ = "Row-major Rows constructor (index_sequence)";

      initialize();

      // Setup of a regular row selection
      {
         auto rs = blaze::rows( mat_, index_sequence<0,3,2>() );

         if( rs.rows() != 3UL || rs.columns() != mat_.columns() ||
             rs(0,0) != mat_(0,0) || rs(0,1) != mat_(0,1) || rs(0,2) != mat_(0,2) || rs(0,3) != mat_(0,3) ||
             rs(1,0) != mat_(3,0) || rs(1,1) != mat_(3,1) || rs(1,2) != mat_(3,2) || rs(1,3) != mat_(3,3) ||
             rs(2,0) != mat_(2,0) || rs(2,1) != mat_(2,1) || rs(2,2) != mat_(2,2) || rs(2,3) != mat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         auto rs = blaze::rows( mat_, index_sequence<4>() );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( mat_, index_sequence<0,3,2>() );
         auto rs2 = blaze::rows( rs1, index_sequence<2,1>() );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( mat_, { 0, 3, 2 } );
         auto rs2 = blaze::rows( rs1, index_sequence<2,1>() );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs1 = blaze::rows( mat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto rs2 = blaze::rows( rs1, index_sequence<2,1>() );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via initializer_list
   //=====================================================================================

   {
      test_ = "Row-major Rows constructor (initializer_list)";

      initialize();

      // Setup of empty row selection
      {
         std::initializer_list<size_t> indices{};
         auto rs = blaze::rows( mat_, indices );

         if( rs.rows() != 0UL || rs.columns() != mat_.columns() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular row selection
      {
         auto rs = blaze::rows( mat_, { 0, 3, 2 } );

         if( rs.rows() != 3UL || rs.columns() != mat_.columns() ||
             rs(0,0) != mat_(0,0) || rs(0,1) != mat_(0,1) || rs(0,2) != mat_(0,2) || rs(0,3) != mat_(0,3) ||
             rs(1,0) != mat_(3,0) || rs(1,1) != mat_(3,1) || rs(1,2) != mat_(3,2) || rs(1,3) != mat_(3,3) ||
             rs(2,0) != mat_(2,0) || rs(2,1) != mat_(2,1) || rs(2,2) != mat_(2,2) || rs(2,3) != mat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         auto rs = blaze::rows( mat_, { 4 } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( mat_, index_sequence<0,3,2>() );
         auto rs2 = blaze::rows( rs1, { 2, 1 } );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( mat_, { 0, 3, 2 } );
         auto rs2 = blaze::rows( rs1, { 2, 1 } );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs1 = blaze::rows( mat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto rs2 = blaze::rows( rs1, { 2, 1 } );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via std::vector
   //=====================================================================================

   {
      test_ = "Row-major Rows constructor (std::vector)";

      initialize();

      // Setup of empty row selection
      {
         std::vector<size_t> indices;
         auto rs = blaze::rows( mat_, indices );

         if( rs.rows() != 0UL || rs.columns() != mat_.columns() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular row selection
      {
         std::vector<size_t> indices{ 0, 3, 2 };
         auto rs = blaze::rows( mat_, indices );

         if( rs.rows() != 3UL || rs.columns() != mat_.columns() ||
             rs(0,0) != mat_(0,0) || rs(0,1) != mat_(0,1) || rs(0,2) != mat_(0,2) || rs(0,3) != mat_(0,3) ||
             rs(1,0) != mat_(3,0) || rs(1,1) != mat_(3,1) || rs(1,2) != mat_(3,2) || rs(1,3) != mat_(3,3) ||
             rs(2,0) != mat_(2,0) || rs(2,1) != mat_(2,1) || rs(2,2) != mat_(2,2) || rs(2,3) != mat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         std::vector<size_t> indices{ 4 };
         auto rs = blaze::rows( mat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( mat_, index_sequence<0,3,2>() );

         const std::vector<size_t> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( mat_, { 0, 3, 2 } );

         const std::vector<size_t> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto rs1 = blaze::rows( mat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::vector<size_t> indices2{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices2 );;

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via std::array
   //=====================================================================================

   {
      test_ = "Row-major Rows constructor (std::array)";

      initialize();

      // Setup of a regular row selection
      {
         std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs = blaze::rows( mat_, indices );

         if( rs.rows() != 3UL || rs.columns() != mat_.columns() ||
             rs(0,0) != mat_(0,0) || rs(0,1) != mat_(0,1) || rs(0,2) != mat_(0,2) || rs(0,3) != mat_(0,3) ||
             rs(1,0) != mat_(3,0) || rs(1,1) != mat_(3,1) || rs(1,2) != mat_(3,2) || rs(1,3) != mat_(3,3) ||
             rs(2,0) != mat_(2,0) || rs(2,1) != mat_(2,1) || rs(2,2) != mat_(2,2) || rs(2,3) != mat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         std::array<size_t,1UL> indices{ 4 };
         auto rs = blaze::rows( mat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( mat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( mat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto rs1 = blaze::rows( mat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices2{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices2 );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via lambda expression
   //=====================================================================================

   {
      test_ = "Row-major Rows constructor (lambda expression)";

      initialize();

      // Setup of empty row selection
      {
         auto rs = blaze::rows( mat_, []( size_t ){ return 0UL; }, 0UL );

         if( rs.rows() != 0UL || rs.columns() != mat_.columns() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular row selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs = blaze::rows( mat_, [indices]( size_t i ){ return indices[i]; }, 3UL );

         if( rs.rows() != 3UL || rs.columns() != mat_.columns() ||
             rs(0,0) != mat_(0,0) || rs(0,1) != mat_(0,1) || rs(0,2) != mat_(0,2) || rs(0,3) != mat_(0,3) ||
             rs(1,0) != mat_(3,0) || rs(1,1) != mat_(3,1) || rs(1,2) != mat_(3,2) || rs(1,3) != mat_(3,3) ||
             rs(2,0) != mat_(2,0) || rs(2,1) != mat_(2,1) || rs(2,2) != mat_(2,2) || rs(2,3) != mat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         auto rs = blaze::rows( mat_, []( size_t ){ return 4UL; }, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( mat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( mat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto rs1 = blaze::rows( mat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices2{ 2, 1 };
         auto rs2 = blaze::rows( rs1, [indices2]( size_t i ){ return indices2[i]; }, 2UL );

         if( rs2.rows() != 2UL || rs2.columns() != mat_.columns() ||
             rs2(0,0) != mat_(2,0) || rs2(0,1) != mat_(2,1) || rs2(0,2) != mat_(2,2) || rs2(0,3) != mat_(2,3) ||
             rs2(1,0) != mat_(3,0) || rs2(1,1) != mat_(3,1) || rs2(1,2) != mat_(3,2) || rs2(1,3) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup of random in-bounds element selection
   //=====================================================================================

   {
      test_ = "Row-major Rows constructor (stress test)";

      initialize();

      for( size_t rep=0UL; rep<100UL; ++rep )
      {
         blaze::DynamicVector<size_t> indices( blaze::rand<size_t>( 1UL, 20UL ) );
         randomize( indices, 0UL, mat_.rows()-1UL );
         auto rs = blaze::rows( mat_, indices.data(), indices.size() );

         for( size_t i=0UL; i<rs.rows(); ++i ) {
            for( size_t j=0UL; j<rs.columns(); ++j ) {
               if( rs(i,j) != mat_(indices[i],j) ) {
                  std::ostringstream oss;
                  oss << " Test: " << test_ << "\n"
                      << " Error: Setup of row selection failed\n"
                      << " Details:\n"
                      << "   Indices:\n" << indices << "\n"
                      << "   Row selection:\n" << rs << "\n"
                      << "   Matrix:\n" << mat_ << "\n";
                  throw std::runtime_error( oss.str() );
               }
            }
         }
      }
   }


   //=====================================================================================
   // Column-major setup via index_sequence
   //=====================================================================================

   {
      test_ = "Column-major Rows constructor (index_sequence)";

      initialize();

      // Setup of a regular row selection
      {
         auto rs = blaze::rows( tmat_, index_sequence<0,3,2>() );

         if( rs.rows() != 3UL || rs.columns() != tmat_.columns() ||
             rs(0,0) != tmat_(0,0) || rs(0,1) != tmat_(0,1) || rs(0,2) != tmat_(0,2) || rs(0,3) != tmat_(0,3) ||
             rs(1,0) != tmat_(3,0) || rs(1,1) != tmat_(3,1) || rs(1,2) != tmat_(3,2) || rs(1,3) != tmat_(3,3) ||
             rs(2,0) != tmat_(2,0) || rs(2,1) != tmat_(2,1) || rs(2,2) != tmat_(2,2) || rs(2,3) != tmat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         auto rs = blaze::rows( tmat_, index_sequence<4>() );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( tmat_, index_sequence<0,3,2>() );
         auto rs2 = blaze::rows( rs1, index_sequence<2,1>() );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( tmat_, { 0, 3, 2 } );
         auto rs2 = blaze::rows( rs1, index_sequence<2,1>() );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs1 = blaze::rows( tmat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto rs2 = blaze::rows( rs1, index_sequence<2,1>() );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via initializer_list
   //=====================================================================================

   {
      test_ = "Column-major Rows constructor (initializer_list)";

      initialize();

      // Setup of empty row selection
      {
         std::initializer_list<size_t> indices{};
         auto rs = blaze::rows( tmat_, indices );

         if( rs.rows() != 0UL || rs.columns() != tmat_.columns() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular row selection
      {
         auto rs = blaze::rows( tmat_, { 0, 3, 2 } );

         if( rs.rows() != 3UL || rs.columns() != tmat_.columns() ||
             rs(0,0) != tmat_(0,0) || rs(0,1) != tmat_(0,1) || rs(0,2) != tmat_(0,2) || rs(0,3) != tmat_(0,3) ||
             rs(1,0) != tmat_(3,0) || rs(1,1) != tmat_(3,1) || rs(1,2) != tmat_(3,2) || rs(1,3) != tmat_(3,3) ||
             rs(2,0) != tmat_(2,0) || rs(2,1) != tmat_(2,1) || rs(2,2) != tmat_(2,2) || rs(2,3) != tmat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         auto rs = blaze::rows( tmat_, { 4 } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( tmat_, index_sequence<0,3,2>() );
         auto rs2 = blaze::rows( rs1, { 2, 1 } );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( tmat_, { 0, 3, 2 } );
         auto rs2 = blaze::rows( rs1, { 2, 1 } );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs1 = blaze::rows( tmat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto rs2 = blaze::rows( rs1, { 2, 1 } );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via std::vector
   //=====================================================================================

   {
      test_ = "Column-major Rows constructor (std::vector)";

      initialize();

      // Setup of empty row selection
      {
         std::vector<size_t> indices;
         auto rs = blaze::rows( tmat_, indices );

         if( rs.rows() != 0UL || rs.columns() != tmat_.columns() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular row selection
      {
         std::vector<size_t> indices{ 0, 3, 2 };
         auto rs = blaze::rows( tmat_, indices );

         if( rs.rows() != 3UL || rs.columns() != tmat_.columns() ||
             rs(0,0) != tmat_(0,0) || rs(0,1) != tmat_(0,1) || rs(0,2) != tmat_(0,2) || rs(0,3) != tmat_(0,3) ||
             rs(1,0) != tmat_(3,0) || rs(1,1) != tmat_(3,1) || rs(1,2) != tmat_(3,2) || rs(1,3) != tmat_(3,3) ||
             rs(2,0) != tmat_(2,0) || rs(2,1) != tmat_(2,1) || rs(2,2) != tmat_(2,2) || rs(2,3) != tmat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         std::vector<size_t> indices{ 4 };
         auto rs = blaze::rows( tmat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( tmat_, index_sequence<0,3,2>() );

         const std::vector<size_t> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( tmat_, { 0, 3, 2 } );

         const std::vector<size_t> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto rs1 = blaze::rows( tmat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::vector<size_t> indices2{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices2 );;

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via std::array
   //=====================================================================================

   {
      test_ = "Column-major Rows constructor (std::array)";

      initialize();

      // Setup of a regular row selection
      {
         std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs = blaze::rows( tmat_, indices );

         if( rs.rows() != 3UL || rs.columns() != tmat_.columns() ||
             rs(0,0) != tmat_(0,0) || rs(0,1) != tmat_(0,1) || rs(0,2) != tmat_(0,2) || rs(0,3) != tmat_(0,3) ||
             rs(1,0) != tmat_(3,0) || rs(1,1) != tmat_(3,1) || rs(1,2) != tmat_(3,2) || rs(1,3) != tmat_(3,3) ||
             rs(2,0) != tmat_(2,0) || rs(2,1) != tmat_(2,1) || rs(2,2) != tmat_(2,2) || rs(2,3) != tmat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         std::array<size_t,1UL> indices{ 4 };
         auto rs = blaze::rows( tmat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( tmat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( tmat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto rs1 = blaze::rows( tmat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices2{ 2, 1 };
         auto rs2 = blaze::rows( rs1, indices2 );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via lambda expression
   //=====================================================================================

   {
      test_ = "Column-major Rows constructor (lambda expression)";

      initialize();

      // Setup of empty row selection
      {
         auto rs = blaze::rows( tmat_, []( size_t ){ return 0UL; }, 0UL );

         if( rs.rows() != 0UL || rs.columns() != tmat_.columns() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular row selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto rs = blaze::rows( tmat_, [indices]( size_t i ){ return indices[i]; }, 3UL );

         if( rs.rows() != 3UL || rs.columns() != tmat_.columns() ||
             rs(0,0) != tmat_(0,0) || rs(0,1) != tmat_(0,1) || rs(0,2) != tmat_(0,2) || rs(0,3) != tmat_(0,3) ||
             rs(1,0) != tmat_(3,0) || rs(1,1) != tmat_(3,1) || rs(1,2) != tmat_(3,2) || rs(1,3) != tmat_(3,3) ||
             rs(2,0) != tmat_(2,0) || rs(2,1) != tmat_(2,1) || rs(2,2) != tmat_(2,2) || rs(2,3) != tmat_(2,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds row selection
      try {
         auto rs = blaze::rows( tmat_, []( size_t ){ return 4UL; }, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a row selection on a compile-time row selection
      {
         auto rs1 = blaze::rows( tmat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an explicit row selection
      {
         auto rs1 = blaze::rows( tmat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto rs2 = blaze::rows( rs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a row selection on an implicit row selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto rs1 = blaze::rows( tmat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices2{ 2, 1 };
         auto rs2 = blaze::rows( rs1, [indices2]( size_t i ){ return indices2[i]; }, 2UL );

         if( rs2.rows() != 2UL || rs2.columns() != tmat_.columns() ||
             rs2(0,0) != tmat_(2,0) || rs2(0,1) != tmat_(2,1) || rs2(0,2) != tmat_(2,2) || rs2(0,3) != tmat_(2,3) ||
             rs2(1,0) != tmat_(3,0) || rs2(1,1) != tmat_(3,1) || rs2(1,2) != tmat_(3,2) || rs2(1,3) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of row selection failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup of random in-bounds element selection
   //=====================================================================================

   {
      test_ = "Column-major Rows constructor (stress test)";

      initialize();

      for( size_t rep=0UL; rep<100UL; ++rep )
      {
         blaze::DynamicVector<size_t> indices( blaze::rand<size_t>( 1UL, 20UL ) );
         randomize( indices, 0UL, tmat_.rows()-1UL );
         auto rs = blaze::rows( tmat_, indices.data(), indices.size() );

         for( size_t i=0UL; i<rs.rows(); ++i ) {
            for( size_t j=0UL; j<rs.columns(); ++j ) {
               if( rs(i,j) != tmat_(indices[i],j) ) {
                  std::ostringstream oss;
                  oss << " Test: " << test_ << "\n"
                      << " Error: Setup of row selection failed\n"
                      << " Details:\n"
                      << "   Indices:\n" << indices << "\n"
                      << "   Row selection:\n" << rs << "\n"
                      << "   Matrix:\n" << tmat_ << "\n";
                  throw std::runtime_error( oss.str() );
               }
            }
         }
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
void SparseSymmetricTest::testFunctionCall()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::operator()";

      initialize();

      RT rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      // Assignment to the element (1,1)
      {
         rs(1,1) = 9;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 9UL );
         checkNonZeros( rs  , 0UL, 3UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 3UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 9UL );

         if( rs(0,0) != 0 || rs(0,1) !=  1 || rs(0,2) != 9 || rs(0,3) != -2 ||
             rs(1,0) != 0 || rs(1,1) !=  9 || rs(1,2) != 3 || rs(1,3) !=  4 ||
             rs(2,0) != 0 || rs(2,1) != -2 || rs(2,2) != 4 || rs(2,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1  9 -2 )\n( 0  9  3  4 )\n( 0 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
             mat_(1,0) != 0 || mat_(1,1) !=  1 || mat_(1,2) != 9 || mat_(1,3) != -2 ||
             mat_(2,0) != 0 || mat_(2,1) !=  9 || mat_(2,2) != 3 || mat_(2,3) !=  4 ||
             mat_(3,0) != 0 || mat_(3,1) != -2 || mat_(3,2) != 4 || mat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  9 -2 )\n"
                                        "( 0  9  3  4 )\n"
                                        "( 0 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (2,1)
      {
         rs(2,1) = 0;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 7UL );
         checkNonZeros( rs  , 0UL, 2UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 9 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 9 || rs(1,2) != 3 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) != 4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1  9  0 )\n( 0  9  3  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) != 0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) != 9 || mat_(1,3) != 0 ||
             mat_(2,0) != 0 || mat_(2,1) != 9 || mat_(2,2) != 3 || mat_(2,3) != 4 ||
             mat_(3,0) != 0 || mat_(3,1) != 0 || mat_(3,2) != 4 || mat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  9  0 )\n"
                                        "( 0  9  3  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,2)
      {
         rs(1,2) = 11;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 7UL );
         checkNonZeros( rs  , 0UL, 2UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) !=  9 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1  9  0 )\n( 0  9 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) !=  9 || mat_(1,3) != 0 ||
             mat_(2,0) != 0 || mat_(2,1) != 9 || mat_(2,2) != 11 || mat_(2,3) != 4 ||
             mat_(3,0) != 0 || mat_(3,1) != 0 || mat_(3,2) !=  4 || mat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  9  0 )\n"
                                        "( 0  9 11  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (0,1)
      {
         rs(0,1) += 3;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 7UL );
         checkNonZeros( rs  , 0UL, 2UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  9 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  9  0 )\n( 0  9 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 4 || mat_(1,2) !=  9 || mat_(1,3) != 0 ||
             mat_(2,0) != 0 || mat_(2,1) != 9 || mat_(2,2) != 11 || mat_(2,3) != 4 ||
             mat_(3,0) != 0 || mat_(3,1) != 0 || mat_(3,2) !=  4 || mat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  9  0 )\n"
                                        "( 0  9 11  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (0,2)
      {
         rs(0,2) -= 6;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 7UL );
         checkNonZeros( rs  , 0UL, 2UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 3 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4 15  0 )\n( 0 15 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 4 || mat_(1,2) !=  3 || mat_(1,3) != 0 ||
             mat_(2,0) != 0 || mat_(2,1) != 3 || mat_(2,2) != 11 || mat_(2,3) != 4 ||
             mat_(3,0) != 0 || mat_(3,1) != 0 || mat_(3,2) !=  4 || mat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  3  0 )\n"
                                        "( 0  3 11  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,2)
      {
         rs(1,2) *= 2;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 7UL );
         checkNonZeros( rs  , 0UL, 2UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 3 || rs(1,2) != 22 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  3  0 )\n( 0  3 22  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 4 || mat_(1,2) !=  3 || mat_(1,3) != 0 ||
             mat_(2,0) != 0 || mat_(2,1) != 3 || mat_(2,2) != 22 || mat_(2,3) != 4 ||
             mat_(3,0) != 0 || mat_(3,1) != 0 || mat_(3,2) !=  4 || mat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  3  0 )\n"
                                        "( 0  3 22  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (1,2)
      {
         rs(1,2) /= 2;

         checkRows    ( rs  , 3UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 7UL );
         checkNonZeros( rs  , 0UL, 2UL );
         checkNonZeros( rs  , 1UL, 3UL );
         checkNonZeros( rs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 3 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  3  0 )\n( 0  3 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 4 || mat_(1,2) !=  3 || mat_(1,3) != 0 ||
             mat_(2,0) != 0 || mat_(2,1) != 3 || mat_(2,2) != 11 || mat_(2,3) != 4 ||
             mat_(3,0) != 0 || mat_(3,1) != 0 || mat_(3,2) !=  4 || mat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  3  0 )\n"
                                        "( 0  3 11  4 )\n"
                                        "( 0  0  4  5 )\n";
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

      ORT rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      // Assignment to the element (1,1)
      {
         rs(1,1) = 9;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 9UL );
         checkNonZeros( rs   , 0UL, 3UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 3UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 9UL );

         if( rs(0,0) != 0 || rs(0,1) !=  1 || rs(0,2) != 9 || rs(0,3) != -2 ||
             rs(1,0) != 0 || rs(1,1) !=  9 || rs(1,2) != 3 || rs(1,3) !=  4 ||
             rs(2,0) != 0 || rs(2,1) != -2 || rs(2,2) != 4 || rs(2,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1  9 -2 )\n( 0  9  3  4 )\n( 0 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) != 0 || tmat_(1,1) !=  1 || tmat_(1,2) != 9 || tmat_(1,3) != -2 ||
             tmat_(2,0) != 0 || tmat_(2,1) !=  9 || tmat_(2,2) != 3 || tmat_(2,3) !=  4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != -2 || tmat_(3,2) != 4 || tmat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  9 -2 )\n"
                                        "( 0  9  3  4 )\n"
                                        "( 0 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (2,1)
      {
         rs(2,1) = 0;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 7UL );
         checkNonZeros( rs   , 0UL, 2UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 9 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 9 || rs(1,2) != 3 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) != 4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1  9  0 )\n( 0  9  3  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != 0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) != 9 || tmat_(1,3) != 0 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 9 || tmat_(2,2) != 3 || tmat_(2,3) != 4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) != 4 || tmat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  9  0 )\n"
                                        "( 0  9  3  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Assignment to the element (1,2)
      {
         rs(1,2) = 11;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 7UL );
         checkNonZeros( rs   , 0UL, 2UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) !=  9 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1  9  0 )\n( 0  9 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  9 || tmat_(1,3) != 0 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 9 || tmat_(2,2) != 11 || tmat_(2,3) != 4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  4 || tmat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  9  0 )\n"
                                        "( 0  9 11  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Addition assignment to the element (0,1)
      {
         rs(0,1) += 3;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 7UL );
         checkNonZeros( rs   , 0UL, 2UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  9 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 9 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  9  0 )\n( 0  9 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 4 || tmat_(1,2) !=  9 || tmat_(1,3) != 0 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 9 || tmat_(2,2) != 11 || tmat_(2,3) != 4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  4 || tmat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  9  0 )\n"
                                        "( 0  9 11  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Subtraction assignment to the element (0,2)
      {
         rs(0,2) -= 6;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 7UL );
         checkNonZeros( rs   , 0UL, 2UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 3 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  3  0 )\n( 0  3 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 4 || tmat_(1,2) !=  3 || tmat_(1,3) != 0 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 3 || tmat_(2,2) != 11 || tmat_(2,3) != 4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  4 || tmat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  3  0 )\n"
                                        "( 0  3 11  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Multiplication assignment to the element (1,2)
      {
         rs(1,2) *= 2;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 7UL );
         checkNonZeros( rs   , 0UL, 2UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 3 || rs(1,2) != 22 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  3  0 )\n( 0  3 22  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 4 || tmat_(1,2) !=  3 || tmat_(1,3) != 0 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 3 || tmat_(2,2) != 22 || tmat_(2,3) != 4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  4 || tmat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  3  0 )\n"
                                        "( 0  3 22  4 )\n"
                                        "( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Division assignment to the element (1,2)
      {
         rs(1,2) /= 2;

         checkRows    ( rs   , 3UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 7UL );
         checkNonZeros( rs   , 0UL, 2UL );
         checkNonZeros( rs   , 1UL, 3UL );
         checkNonZeros( rs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( rs(0,0) != 0 || rs(0,1) != 4 || rs(0,2) !=  3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 3 || rs(1,2) != 11 || rs(1,3) != 4 ||
             rs(2,0) != 0 || rs(2,1) != 0 || rs(2,2) !=  4 || rs(2,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  4  3  0 )\n( 0  3 11  4 )\n( 0  0  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 4 || tmat_(1,2) !=  3 || tmat_(1,3) != 0 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 3 || tmat_(2,2) != 11 || tmat_(2,3) != 4 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 0 || tmat_(3,2) !=  4 || tmat_(3,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  4  3  0 )\n"
                                        "( 0  3 11  4 )\n"
                                        "( 0  0  4  5 )\n";
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
void SparseSymmetricTest::testIterator()
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

         RT rs = blaze::rows( mat_, { 2UL } );
         RT::ConstIterator it( begin( rs, 0UL ) );

         if( it == end( rs, 0UL ) || it->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         RT rs = blaze::rows( mat_, { 1UL } );
         const ptrdiff_t number( end( rs, 0UL ) - begin( rs, 0UL ) );

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

      // Counting the number of elements in 2nd row via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         RT rs = blaze::rows( mat_, { 2UL } );
         const ptrdiff_t number( cend( rs, 0UL ) - cbegin( rs, 0UL ) );

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

         RT rs = blaze::rows( mat_, { 2UL } );
         RT::ConstIterator it ( cbegin( rs, 0UL ) );
         RT::ConstIterator end( cend( rs, 0UL ) );

         if( it == end || it->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || it->value() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Row-major assignment via Iterator";

         RT rs = blaze::rows( mat_, { 3UL } );
         int value = 6;

         for( RT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it = value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 6 || rs(0,2) != 7 || rs(0,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 6 7 8 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) != 0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) != 0 || mat_(1,3) != 6 ||
             mat_(2,0) != 0 || mat_(2,1) != 0 || mat_(2,2) != 3 || mat_(2,3) != 7 ||
             mat_(3,0) != 0 || mat_(3,1) != 6 || mat_(3,2) != 7 || mat_(3,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  6 )\n"
                                        "( 0  0  3  7 )\n"
                                        "( 0  6  7  8 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         RT rs = blaze::rows( mat_, { 3UL } );
         int value = 2;

         for( RT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it += value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 8 || rs(0,2) != 10 || rs(0,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 8 10 12 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) !=  0 || mat_(1,3) !=  8 ||
             mat_(2,0) != 0 || mat_(2,1) != 0 || mat_(2,2) !=  3 || mat_(2,3) != 10 ||
             mat_(3,0) != 0 || mat_(3,1) != 8 || mat_(3,2) != 10 || mat_(3,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  8 )\n"
                                        "( 0  0  3 10 )\n"
                                        "( 0  8 10 12 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         RT rs = blaze::rows( mat_, { 3UL } );
         int value = 2;

         for( RT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 6 || rs(0,2) != 7 || rs(0,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 6 7 8 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) != 0 || mat_(0,3) != 0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) != 0 || mat_(1,3) != 6 ||
             mat_(2,0) != 0 || mat_(2,1) != 0 || mat_(2,2) != 3 || mat_(2,3) != 7 ||
             mat_(3,0) != 0 || mat_(3,1) != 6 || mat_(3,2) != 7 || mat_(3,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  6 )\n"
                                        "( 0  0  3  7 )\n"
                                        "( 0  6  7  8 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         RT rs = blaze::rows( mat_, { 3UL } );
         int value = 1;

         for( RT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 6 || rs(0,2) != 14 || rs(0,3) != 24 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 6 14 24 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) !=  0 || mat_(0,3) !=  0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) !=  0 || mat_(1,3) !=  6 ||
             mat_(2,0) != 0 || mat_(2,1) != 0 || mat_(2,2) !=  3 || mat_(2,3) != 14 ||
             mat_(3,0) != 0 || mat_(3,1) != 6 || mat_(3,2) != 14 || mat_(3,3) != 24 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  6 )\n"
                                        "( 0  0  3 14 )\n"
                                        "( 0  6 14 24 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         RT rs = blaze::rows( mat_, { 3UL } );

         for( RT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( rs(0,0) != 0 || rs(0,1) != 3 || rs(0,2) != 7 || rs(0,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 3 7 12 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 0 || mat_(0,1) != 0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
             mat_(1,0) != 0 || mat_(1,1) != 1 || mat_(1,2) != 0 || mat_(1,3) !=  3 ||
             mat_(2,0) != 0 || mat_(2,1) != 0 || mat_(2,2) != 3 || mat_(2,3) !=  7 ||
             mat_(3,0) != 0 || mat_(3,1) != 3 || mat_(3,2) != 7 || mat_(3,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  3 )\n"
                                        "( 0  0  3  7 )\n"
                                        "( 0  3  7 12 )\n";
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

         ORT rs = blaze::rows( tmat_, { 2UL } );
         ORT::ConstIterator it( begin( rs, 0UL ) );

         if( it == end( rs, 0UL ) || it->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st row via Iterator (end-begin)
      {
         test_ = "Column-major Iterator subtraction (end-begin)";

         ORT rs = blaze::rows( tmat_, { 1UL } );
         const ptrdiff_t number( end( rs, 0UL ) - begin( rs, 0UL ) );

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

      // Counting the number of elements in 2nd row via ConstIterator (end-begin)
      {
         test_ = "Column-major ConstIterator subtraction (end-begin)";

         ORT rs = blaze::rows( tmat_, { 2UL } );
         const ptrdiff_t number( cend( rs, 0UL ) - cbegin( rs, 0UL ) );

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

         ORT rs = blaze::rows( tmat_, { 2UL } );
         ORT::ConstIterator it ( cbegin( rs, 0UL ) );
         ORT::ConstIterator end( cend( rs, 0UL ) );

         if( it == end || it->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || it->value() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator pre-increment failed\n";
            throw std::runtime_error( oss.str() );
         }

         it++;

         if( it != end ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator post-increment failed\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing assignment via Iterator
      {
         test_ = "Column-major assignment via Iterator";

         ORT rs = blaze::rows( tmat_, { 3UL } );
         int value = 6;

         for( ORT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it = value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 6 || rs(0,2) != 7 || rs(0,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 6 7 8 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != 0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) != 0 || tmat_(1,3) != 6 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != 3 || tmat_(2,3) != 7 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 6 || tmat_(3,2) != 7 || tmat_(3,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  6 )\n"
                                        "( 0  0  3  7 )\n"
                                        "( 0  6  7  8 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Column-major addition assignment via Iterator";

         ORT rs = blaze::rows( tmat_, { 3UL } );
         int value = 2;

         for( ORT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it += value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 8 || rs(0,2) != 10 || rs(0,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 8 10 12 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  8 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) !=  3 || tmat_(2,3) != 10 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 8 || tmat_(3,2) != 10 || tmat_(3,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  8 )\n"
                                        "( 0  0  3 10 )\n"
                                        "( 0  8 10 12 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Column-major subtraction assignment via Iterator";

         ORT rs = blaze::rows( tmat_, { 3UL } );
         int value = 2;

         for( ORT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 6 || rs(0,2) != 7 || rs(0,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 6 7 8 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != 0 || tmat_(0,3) != 0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) != 0 || tmat_(1,3) != 6 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != 3 || tmat_(2,3) != 7 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 6 || tmat_(3,2) != 7 || tmat_(3,3) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  6 )\n"
                                        "( 0  0  3  7 )\n"
                                        "( 0  6  7  8 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Column-major multiplication assignment via Iterator";

         ORT rs = blaze::rows( tmat_, { 3UL } );
         int value = 1;

         for( ORT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( rs(0,0) != 0 || rs(0,1) != 6 || rs(0,2) != 14 || rs(0,3) != 24 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 6 14 24 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) !=  0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) !=  0 || tmat_(1,3) !=  6 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) !=  3 || tmat_(2,3) != 14 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 6 || tmat_(3,2) != 14 || tmat_(3,3) != 24 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  6 )\n"
                                        "( 0  0  3 14 )\n"
                                        "( 0  6 14 24 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Column-major division assignment via Iterator";

         ORT rs = blaze::rows( tmat_, { 3UL } );

         for( ORT::Iterator it=begin( rs, 0UL ); it!=end( rs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( rs(0,0) != 0 || rs(0,1) != 3 || rs(0,2) != 7 || rs(0,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 3 7 12 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 0 || tmat_(0,1) != 0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
             tmat_(1,0) != 0 || tmat_(1,1) != 1 || tmat_(1,2) != 0 || tmat_(1,3) !=  3 ||
             tmat_(2,0) != 0 || tmat_(2,1) != 0 || tmat_(2,2) != 3 || tmat_(2,3) !=  7 ||
             tmat_(3,0) != 0 || tmat_(3,1) != 3 || tmat_(3,2) != 7 || tmat_(3,3) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n"
                                        "( 0  1  0  3 )\n"
                                        "( 0  0  3  7 )\n"
                                        "( 0  3  7 12 )\n";
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
void SparseSymmetricTest::testNonZeros()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::nonZeros()";

      initialize();

      // Initialization check
      RT rs = blaze::rows( mat_, { 1UL, 2UL } );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 4UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 2UL );

      if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != -2 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 3 || rs(1,3) !=  4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  3  4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the row selection
      rs(1,2) = 0;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 1UL );

      if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != -2 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) !=  4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  0  4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      mat_(2,3) = 5;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 1UL );

      if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != -2 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  0  5 )\n";
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
      ORT rs = blaze::rows( tmat_, { 1UL, 2UL } );

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 4UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 2UL );

      if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != -2 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 3 || rs(1,3) !=  4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  3  4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the row selection
      rs(1,2) = 0;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 1UL );

      if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != -2 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) !=  4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  0  4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      tmat_(2,3) = 5;

      checkRows    ( rs, 2UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 3UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 1UL );

      if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != 0 || rs(0,3) != -2 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  0  5 )\n";
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
void SparseSymmetricTest::testReset()
{
   //=====================================================================================
   // Row-major single element reset
   //=====================================================================================

   {
      test_ = "Row-major reset() function";

      using blaze::reset;
      using blaze::isDefault;

      initialize();

      RT rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      reset( rs(0,1) );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 6UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0 -2 )\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) != -2 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 3 || mat_(2,3) !=  4 ||
          mat_(3,0) != 0 || mat_(3,1) != -2 || mat_(3,2) != 4 || mat_(3,3) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0 -2 )\n"
                                     "( 0  0  3  4 )\n"
                                     "( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major reset
   //=====================================================================================

   {
      test_ = "Row-major Rows::reset() (lvalue)";

      initialize();

      RT rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      reset( rs );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 0UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 0UL );

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
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Rows::reset() (rvalue)";

      initialize();

      reset( blaze::rows( mat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 0UL );

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 0 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
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

      ORT rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      reset( rs(0,1) );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 6UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0 -2 )\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) != -2 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 3 || tmat_(2,3) !=  4 ||
          tmat_(3,0) != 0 || tmat_(3,1) != -2 || tmat_(3,2) != 4 || tmat_(3,3) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0 -2 )\n"
                                     "( 0  0  3  4 )\n"
                                     "( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major reset
   //=====================================================================================

   {
      test_ = "Column-major Rows::reset() (lvalue)";

      initialize();

      ORT rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      reset( rs );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 0UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 0UL );

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
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major Rows::reset() (rvalue)";

      initialize();

      reset( blaze::rows( tmat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 0UL );

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 0 || tmat_(2,3) !=  0 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
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
void SparseSymmetricTest::testClear()
{
   //=====================================================================================
   // Row-major single element clear
   //=====================================================================================

   {
      test_ = "Row-major clear() function";

      using blaze::clear;
      using blaze::isDefault;

      initialize();

      RT rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      clear( rs(0,1) );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 6UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0 -2 )\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) != -2 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 3 || mat_(2,3) !=  4 ||
          mat_(3,0) != 0 || mat_(3,1) != -2 || mat_(3,2) != 4 || mat_(3,3) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0 -2 )\n"
                                     "( 0  0  3  4 )\n"
                                     "( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major clear
   //=====================================================================================

   {
      test_ = "Row-major Rows::clear() (lvalue)";

      initialize();

      RT rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      clear( rs );

      checkRows    ( rs  , 3UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 0UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 0UL );

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
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Row-major Rows::clear() (rvalue)";

      initialize();

      clear( blaze::rows( mat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 0UL );

      if( mat_(0,0) != 0 || mat_(0,1) !=  0 || mat_(0,2) != 0 || mat_(0,3) !=  0 ||
          mat_(1,0) != 0 || mat_(1,1) !=  0 || mat_(1,2) != 0 || mat_(1,3) !=  0 ||
          mat_(2,0) != 0 || mat_(2,1) !=  0 || mat_(2,2) != 0 || mat_(2,3) !=  0 ||
          mat_(3,0) != 0 || mat_(3,1) !=  0 || mat_(3,2) != 0 || mat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
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

      ORT rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      clear( rs(0,1) );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 6UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( !isDefault( rs(0,1) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0 -2 )\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) != -2 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 3 || tmat_(2,3) !=  4 ||
          tmat_(3,0) != 0 || tmat_(3,1) != -2 || tmat_(3,2) != 4 || tmat_(3,3) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0 -2 )\n"
                                     "( 0  0  3  4 )\n"
                                     "( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major clear
   //=====================================================================================

   {
      test_ = "Column-major Rows::clear() (lvalue)";

      initialize();

      ORT rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      clear( rs );

      checkRows    ( rs   , 3UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 0UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 0UL );

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
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   {
      test_ = "Column-major Rows::clear() (rvalue)";

      initialize();

      clear( blaze::rows( tmat_, { 1UL, 2UL, 3UL } ) );

      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 0UL );

      if( tmat_(0,0) != 0 || tmat_(0,1) !=  0 || tmat_(0,2) != 0 || tmat_(0,3) !=  0 ||
          tmat_(1,0) != 0 || tmat_(1,1) !=  0 || tmat_(1,2) != 0 || tmat_(1,3) !=  0 ||
          tmat_(2,0) != 0 || tmat_(2,1) !=  0 || tmat_(2,2) != 0 || tmat_(2,3) !=  0 ||
          tmat_(3,0) != 0 || tmat_(3,1) !=  0 || tmat_(3,2) != 0 || tmat_(3,3) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n"
                                     "( 0  0  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reserve() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reserve() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testReserve()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::reserve()";

      MT mat( 20UL );

      RT rs = blaze::rows( mat, { 1UL } );

      // Increasing the capacity of the row selection
      rs.reserve( 10UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 10UL );
      checkNonZeros( rs,  0UL );

      // Further increasing the capacity of the row selection
      rs.reserve( 20UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 20UL );
      checkNonZeros( rs,  0UL );
   }

   {
      test_ = "Row-major Rows::reserve( size_t )";

      MT mat( 20UL );

      RT rs = blaze::rows( mat, { 1UL } );

      // Increasing the capacity of a single row
      rs.reserve( 0UL, 10UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 10UL );
      checkNonZeros( rs,  0UL );

      // Further increasing the capacity of a single row
      rs.reserve( 0UL, 15UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 15UL );
      checkNonZeros( rs,  0UL );
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::reserve()";

      OMT mat( 20UL );

      ORT rs = blaze::rows( mat, { 1UL } );

      // Increasing the capacity of the row selection
      rs.reserve( 10UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 10UL );
      checkNonZeros( rs,  0UL );

      // Further increasing the capacity of the row selection
      rs.reserve( 20UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 20UL );
      checkNonZeros( rs,  0UL );
   }

   {
      test_ = "Column-major Rows::reserve( size_t )";

      OMT mat( 20UL );

      ORT rs = blaze::rows( mat, { 1UL } );

      // Increasing the capacity of a single row
      rs.reserve( 0UL, 10UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 10UL );
      checkNonZeros( rs,  0UL );

      // Further increasing the capacity of a single row
      rs.reserve( 0UL, 15UL );

      checkRows    ( rs,  1UL );
      checkColumns ( rs, 20UL );
      checkCapacity( rs, 15UL );
      checkNonZeros( rs,  0UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c trim() member functions of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c trim() member functions of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testTrim()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::trim()";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Increasing the row capacity of the matrix
      rs.reserve( 0UL, 10UL );
      rs.reserve( 1UL, 20UL );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkCapacity( rs  , 30UL );
      checkCapacity( rs  ,  0UL, 10UL );
      checkCapacity( rs  ,  1UL, 20UL );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, 10UL );
      checkCapacity( mat_,  3UL, 20UL );

      // Trimming the matrix
      rs.trim();

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkCapacity( rs  , 30UL );
      checkCapacity( rs  ,  0UL, rs.nonZeros( 0UL ) );
      checkCapacity( rs  ,  1UL, rs.nonZeros( 1UL ) );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, mat_.nonZeros( 2UL ) );
      checkCapacity( mat_,  3UL, mat_.nonZeros( 3UL ) );
   }

   {
      test_ = "Row-major Rows::trim( size_t )";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Increasing the row capacity of the matrix
      rs.reserve( 0UL, 10UL );
      rs.reserve( 1UL, 20UL );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkCapacity( rs  , 30UL );
      checkCapacity( rs  ,  0UL, 10UL );
      checkCapacity( rs  ,  1UL, 20UL );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, 10UL );
      checkCapacity( mat_,  3UL, 20UL );

      // Trimming the 0th row
      rs.trim( 0UL );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkCapacity( rs  , 30UL );
      checkCapacity( rs  ,  0UL, rs.nonZeros( 0UL ) );
      checkCapacity( rs  ,  1UL, 30UL - rs.nonZeros( 0UL ) );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, mat_.nonZeros( 2UL ) );
      checkCapacity( mat_,  3UL, 30UL - mat_.nonZeros( 2UL ) );

      // Trimming the 1st row
      rs.trim( 1UL );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkCapacity( rs  , 30UL );
      checkCapacity( rs  ,  0UL, rs.nonZeros( 0UL ) );
      checkCapacity( rs  ,  1UL, rs.nonZeros( 1UL ) );
      checkCapacity( mat_, 30UL );
      checkCapacity( mat_,  2UL, mat_.nonZeros( 2UL ) );
      checkCapacity( mat_,  3UL, mat_.nonZeros( 3UL ) );
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   // No column-major matrix test required
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c set() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c set() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testSet()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::set()";

      initialize();

      RT rs = blaze::rows( mat_, { 0UL, 1UL } );

      // Setting a non-zero element at the end of the 0th row
      rs.set( 0UL, 3UL, 1 );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the beginning of the 0th row
      rs.set( 0UL, 0UL, 2 );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  4UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != 2 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the center of the 0th row
      rs.set( 0UL, 1UL, 3 );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  6UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( rs(0,0) != 2 || rs(0,1) != 3 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 3 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 3  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting an already existing element
      rs.set( 1UL, 1UL, 4 );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  6UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( rs(0,0) != 2 || rs(0,1) != 3 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 3 || rs(1,1) != 4 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 3  4  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::set()";

      initialize();

      ORT rs = blaze::rows( tmat_, { 0UL, 1UL } );

      // Setting a non-zero element at the end of the 0th row
      rs.set( 0UL, 3UL, 1 );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 9UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the beginning of the 0th row
      rs.set( 0UL, 0UL, 2 );

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != 2 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the center of the 0th row
      rs.set( 0UL, 1UL, 3 );

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  6UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 12UL );

      if( rs(0,0) != 2 || rs(0,1) != 3 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 3 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 3  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting an already existing element
      rs.set( 1UL, 1UL, 4 );

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  6UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 12UL );

      if( rs(0,0) != 2 || rs(0,1) != 3 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 3 || rs(1,1) != 4 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 3  4  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c insert() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c insert() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testInsert()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::insert()";

      initialize();

      RT rs = blaze::rows( mat_, { 0UL, 1UL } );

      // Inserting a non-zero element at the end of the 0th row
      rs.insert( 0UL, 3UL, 1 );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the beginning of the 0th row
      rs.insert( 0UL, 0UL, 2 );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  4UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( rs(0,0) != 2 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the center of the 0th row
      rs.insert( 0UL, 1UL, 3 );

      checkRows    ( rs  ,  2UL );
      checkColumns ( rs  ,  4UL );
      checkNonZeros( rs  ,  6UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( rs(0,0) != 2 || rs(0,1) != 3 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 3 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to insert an already existing element
      try {
         rs.insert( 1UL, 1UL, 4 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting an existing element succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::insert()";

      initialize();

      ORT rs = blaze::rows( tmat_, { 0UL, 1UL } );

      // Inserting a non-zero element at the end of the 0th row
      rs.insert( 0UL, 3UL, 1 );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 9UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the beginning of the 0th row
      rs.insert( 0UL, 0UL, 2 );

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( rs(0,0) != 2 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 0 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  0  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the center of the 0th row
      rs.insert( 0UL, 1UL, 3 );

      checkRows    ( rs   ,  2UL );
      checkColumns ( rs   ,  4UL );
      checkNonZeros( rs   ,  6UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 12UL );

      if( rs(0,0) != 2 || rs(0,1) != 3 || rs(0,2) != 0 || rs(0,3) !=  1 ||
          rs(1,0) != 3 || rs(1,1) != 1 || rs(1,2) != 0 || rs(1,3) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to insert an already existing element
      try {
         rs.insert( 1UL, 1UL, 4 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting an existing element succeeded\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 2  3  0  1 )\n( 0  1  0 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c append() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c append() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testAppend()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::append()";

      // Appending with pre-allocation in each row
      {
         mat_.reset();

         // Initialization check
         RT rs = blaze::rows( mat_, { 3UL, 2UL, 1UL, 0UL } );
         rs.reserve( 0UL, 2UL );
         rs.reserve( 1UL, 1UL );
         rs.reserve( 2UL, 2UL );
         rs.reserve( 3UL, 3UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 0UL );
         checkNonZeros( rs, 0UL, 0UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 0UL );
         checkNonZeros( rs, 3UL, 0UL );

         // Appending one non-zero element
         rs.append( 2UL, 1UL, 1 );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 1UL );
         checkNonZeros( rs, 0UL, 0UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 1UL );
         checkNonZeros( rs, 3UL, 0UL );

         if( rs(2,1) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.append( 0UL, 0UL, 2 );
         rs.append( 0UL, 3UL, 3 );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 4UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 1UL );
         checkNonZeros( rs, 3UL, 1UL );

         if( rs(2,1) != 1 || rs(0,0) != 2 || rs(3,3) != 2 || rs(0,3) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 2 0 0 3 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 0 0 2 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.append( 3UL, 1UL, 4 );
         rs.append( 3UL, 2UL, 5 );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 8UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 1UL );
         checkNonZeros( rs, 2UL, 2UL );
         checkNonZeros( rs, 3UL, 3UL );

         if( rs(2,1) != 1 || rs(0,0) != 2 || rs(3,3) != 2 || rs(0,3) != 3 ||
             rs(3,1) != 4 || rs(2,0) != 4 || rs(3,2) != 5 || rs(1,0) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 2 0 0 3 )\n( 5 0 0 0 )\n( 4 1 0 0 )\n( 0 4 5 2 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Appending with row finalization
      {
         mat_.reset();

         // Initialization check
         RT rs = blaze::rows( mat_, { 3UL, 2UL, 1UL, 0UL } );
         rs.reserve( 0UL, 2UL );
         rs.reserve( 1UL, 2UL );
         rs.reserve( 2UL, 3UL );
         rs.reserve( 3UL, 2UL );

         // Appending one non-zero element
         rs.append( 0UL, 1UL, 1 );
         rs.finalize( 0UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 9UL );
         checkNonZeros( rs, 2UL );
         checkNonZeros( rs, 0UL, 1UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 1UL );
         checkNonZeros( rs, 3UL, 0UL );

         if( rs(0,1) != 1 || rs(2,3) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n( 0 0 0 0 )\n( 0 0 0 1 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.append( 1UL, 1UL, 2 );
         rs.append( 1UL, 3UL, 3 );
         rs.finalize( 1UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 9UL );
         checkNonZeros( rs, 6UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 2UL );
         checkNonZeros( rs, 2UL, 2UL );
         checkNonZeros( rs, 3UL, 0UL );

         if( rs(0,1) != 1 || rs(2,3) != 1 || rs(1,1) != 2 ||
             rs(2,2) != 2 || rs(1,3) != 3 || rs(0,2) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 1 3 0 )\n( 0 2 0 3 )\n( 0 0 2 1 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.finalize( 2UL );
         rs.append( 3UL, 0UL, 4 );
         rs.append( 3UL, 1UL, 5 );
         rs.finalize( 3UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 9UL );
         checkNonZeros( rs, 9UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 2UL );
         checkNonZeros( rs, 2UL, 3UL );
         checkNonZeros( rs, 3UL, 2UL );

         if( rs(0,1) != 1 || rs(2,3) != 1 || rs(1,1) != 2 || rs(2,2) != 2 ||
             rs(1,3) != 3 || rs(0,2) != 3 || rs(3,0) != 4 || rs(2,0) != 5 || rs(3,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 1 3 0 )\n( 0 2 0 3 )\n( 5 0 2 1 )\n( 4 5 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::append()";

      // Appending with pre-allocation in each row
      {
         tmat_.reset();

         // Initialization check
         ORT rs = blaze::rows( tmat_, { 3UL, 2UL, 1UL, 0UL } );
         rs.reserve( 0UL, 2UL );
         rs.reserve( 1UL, 1UL );
         rs.reserve( 2UL, 2UL );
         rs.reserve( 3UL, 3UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 0UL );
         checkNonZeros( rs, 0UL, 0UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 0UL );
         checkNonZeros( rs, 3UL, 0UL );

         // Appending one non-zero element
         rs.append( 2UL, 1UL, 1 );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 1UL );
         checkNonZeros( rs, 0UL, 0UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 1UL );
         checkNonZeros( rs, 3UL, 0UL );

         if( rs(2,1) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.append( 0UL, 0UL, 2 );
         rs.append( 0UL, 3UL, 3 );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 4UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 1UL );
         checkNonZeros( rs, 3UL, 1UL );

         if( rs(2,1) != 1 || rs(0,0) != 2 || rs(3,3) != 2 || rs(0,3) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 2 0 0 3 )\n( 0 0 0 0 )\n( 0 1 0 0 )\n( 0 0 0 2 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.append( 3UL, 1UL, 4 );
         rs.append( 3UL, 2UL, 5 );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 8UL );
         checkNonZeros( rs, 8UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 1UL );
         checkNonZeros( rs, 2UL, 2UL );
         checkNonZeros( rs, 3UL, 3UL );

         if( rs(2,1) != 1 || rs(0,0) != 2 || rs(3,3) != 2 || rs(0,3) != 3 ||
             rs(3,1) != 4 || rs(2,0) != 4 || rs(3,2) != 5 || rs(1,0) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 2 0 0 3 )\n( 5 0 0 0 )\n( 4 1 0 0 )\n( 0 4 5 2 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Appending with row finalization
      {
         tmat_.reset();

         // Initialization check
         ORT rs = blaze::rows( tmat_, { 3UL, 2UL, 1UL, 0UL } );
         rs.reserve( 0UL, 2UL );
         rs.reserve( 1UL, 2UL );
         rs.reserve( 2UL, 3UL );
         rs.reserve( 3UL, 2UL );

         // Appending one non-zero element
         rs.append( 0UL, 1UL, 1 );
         rs.finalize( 0UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 9UL );
         checkNonZeros( rs, 2UL );
         checkNonZeros( rs, 0UL, 1UL );
         checkNonZeros( rs, 1UL, 0UL );
         checkNonZeros( rs, 2UL, 1UL );
         checkNonZeros( rs, 3UL, 0UL );

         if( rs(0,1) != 1 || rs(2,3) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 1 0 0 )\n( 0 0 0 0 )\n( 0 0 0 1 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.append( 1UL, 1UL, 2 );
         rs.append( 1UL, 3UL, 3 );
         rs.finalize( 1UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 9UL );
         checkNonZeros( rs, 6UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 2UL );
         checkNonZeros( rs, 2UL, 2UL );
         checkNonZeros( rs, 3UL, 0UL );

         if( rs(0,1) != 1 || rs(2,3) != 1 || rs(1,1) != 2 ||
             rs(2,2) != 2 || rs(1,3) != 3 || rs(0,2) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 1 3 0 )\n( 0 2 0 3 )\n( 0 0 2 1 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         rs.finalize( 2UL );
         rs.append( 3UL, 0UL, 4 );
         rs.append( 3UL, 1UL, 5 );
         rs.finalize( 3UL );

         checkRows    ( rs, 4UL );
         checkColumns ( rs, 4UL );
         checkCapacity( rs, 9UL );
         checkNonZeros( rs, 9UL );
         checkNonZeros( rs, 0UL, 2UL );
         checkNonZeros( rs, 1UL, 2UL );
         checkNonZeros( rs, 2UL, 3UL );
         checkNonZeros( rs, 3UL, 2UL );

         if( rs(0,1) != 1 || rs(2,3) != 1 || rs(1,1) != 2 || rs(2,2) != 2 ||
             rs(1,3) != 3 || rs(0,2) != 3 || rs(3,0) != 4 || rs(2,0) != 5 || rs(3,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 1 3 0 )\n( 0 2 0 3 )\n( 5 0 2 1 )\n( 4 5 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c erase() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c erase() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testErase()
{
   //=====================================================================================
   // Row-major index-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Rows::erase( size_t )";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st row
      rs.erase( 1UL, 3UL );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 4UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 3 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st row
      rs.erase( 1UL, 1UL );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  3  4 )\n( 0  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase an already erased element
      rs.erase( 1UL, 3UL );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  3  4 )\n( 0  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major iterator-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Rows::erase( size_t, Iterator )";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st row
      {
         RT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 3UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 4UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 6UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 3 || rs(0,3) != 4 ||
             rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  4 )\n( 0 -2  4  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st row
      {
         RT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 1UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 3UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 4UL );

         if( pos->value() != 4 || pos->index() != 2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 4\n"
                << "   Expected index: 2\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 4 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 4 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  4 )\n( 0  0  4  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st row
      {
         RT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 2UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 1UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 2UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  0 )\n( 0  0  0  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an already erased element
      {
         RT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 3UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 1UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 2UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  0 )\n( 0  0  0  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major iterator-range-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Rows::erase( size_t, Iterator, Iterator )";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Erasing the 0th row
      {
         RT::Iterator pos = rs.erase( 0UL, rs.begin( 0UL ), rs.end( 0UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 4UL );

         if( pos != rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 0 || rs(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th row failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n( 0 -2  0  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the first half of the 1st row
      {
         RT::Iterator pos = rs.erase( 1UL, rs.begin( 1UL ), rs.find( 1UL, 3UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 1UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 2UL );

         if( pos->value() != 5 || pos->index() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 5\n"
                << "   Expected index: 3\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the first half of the 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n( 0  0  0  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the second half of the 1st row
      {
         RT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 3UL ), rs.end( 1UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 0UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 1UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the second half of the 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an empty range
      {
         RT::Iterator pos = rs.erase( 1UL, rs.begin( 1UL ), rs.begin( 1UL ) );

         checkRows    ( rs  , 2UL );
         checkColumns ( rs  , 4UL );
         checkNonZeros( rs  , 0UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 1UL );

         if( pos != rs.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the given end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing an empty range failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major erase function with predicate
   //=====================================================================================

   {
      test_ = "Row-major Rows::erase( Predicate )";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      rs.erase( []( int value ) { return value == 3 || value == 5; } );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 5UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase all elements with value 1
      rs.erase( []( int value ){ return value == 1; } );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 5UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing all elements with value 1 failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major iterator-range-based erase function with predicate
   //=====================================================================================

   {
      test_ = "Row-major Rows::erase( size_t, Iterator, Iterator, Predicate )";

      initialize();

      RT rs = blaze::rows( mat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      rs.erase( 0UL, rs.begin( 0UL ), rs.find( 0UL, 3UL ),
                []( int value ) { return value == 3; } );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 4UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase from an empty range
      rs.erase( 1UL, rs.begin( 1UL ), rs.begin( 1UL ), []( int ){ return true; } );

      checkRows    ( rs  , 2UL );
      checkColumns ( rs  , 4UL );
      checkNonZeros( rs  , 4UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing from an empty range failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major index-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Rows::erase( size_t )";

      initialize();

      ORT rs = blaze::rows( tmat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st row
      rs.erase( 1UL, 3UL );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 3 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st row
      rs.erase( 1UL, 1UL );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  3  4 )\n( 0  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase an already erased element
      rs.erase( 1UL, 3UL );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  3  4 )\n( 0  0  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major iterator-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Rows::erase( size_t, Iterator )";

      initialize();

      ORT rs = blaze::rows( tmat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st row
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 3UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 4UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 6UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 3 || rs(0,3) != 4 ||
             rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  4 )\n( 0 -2  4  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st row
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 1UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 3UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 4UL );

         if( pos->value() != 4 || pos->index() != 2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 4\n"
                << "   Expected index: 2\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 4 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 4 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  4 )\n( 0  0  4  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st row
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 2UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 1UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 2UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  0 )\n( 0  0  0  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an already erased element
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 3UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 1UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 2UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 3 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  3  0 )\n( 0  0  0  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major iterator-range-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Rows::erase( size_t, Iterator, Iterator )";

      initialize();

      ORT rs = blaze::rows( tmat_, { 2UL, 3UL } );

      // Erasing the 0th row
      {
         ORT::Iterator pos = rs.erase( 0UL, rs.begin( 0UL ), rs.end( 0UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 4UL );

         if( pos != rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 0 || rs(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th row failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n( 0 -2  0  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the first half of the 1st row
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.begin( 1UL ), rs.find( 1UL, 3UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 1UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 2UL );

         if( pos->value() != 5 || pos->index() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Value: " << pos->value() << "\n"
                << "   Index: " << pos->index() << "\n"
                << "   Expected value: 5\n"
                << "   Expected index: 3\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the first half of the 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  0  0  0 )\n( 0  0  0  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the second half of the 1st row
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.find( 1UL, 3UL ), rs.end( 1UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 1UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the second half of the 1st row failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an empty range
      {
         ORT::Iterator pos = rs.erase( 1UL, rs.begin( 1UL ), rs.begin( 1UL ) );

         checkRows    ( rs   , 2UL );
         checkColumns ( rs   , 4UL );
         checkNonZeros( rs   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 1UL );

         if( pos != rs.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the given end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs(0,0) != 0 || rs(0,1) != 0 || rs(0,2) != 0 || rs(0,3) != 0 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) != 0 || rs(1,3) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing an empty range failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major erase function with predicate
   //=====================================================================================

   {
      test_ = "Column-major Rows::erase( Predicate )";

      initialize();

      ORT rs = blaze::rows( tmat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      rs.erase( []( int value ) { return value == 3 || value == 5; } );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 5UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase all elements with value 1
      rs.erase( []( int value ){ return value == 1; } );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 5UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing all elements with value 1 failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major iterator-range-based erase function with predicate
   //=====================================================================================

   {
      test_ = "Column-major Rows::erase( size_t, Iterator, Iterator, Predicate )";

      initialize();

      ORT rs = blaze::rows( tmat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      rs.erase( 0UL, rs.begin( 0UL ), rs.find( 0UL, 3UL ),
                []( int value ) { return value == 3; } );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase from an empty range
      rs.erase( 1UL, rs.begin( 1UL ), rs.begin( 1UL ), []( int ){ return true; } );

      checkRows    ( rs   , 2UL );
      checkColumns ( rs   , 4UL );
      checkNonZeros( rs   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( rs(0,0) != 0 || rs(0,1) !=  0 || rs(0,2) != 0 || rs(0,3) != 4 ||
          rs(1,0) != 0 || rs(1,1) != -2 || rs(1,2) != 4 || rs(1,3) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing from an empty range failed\n"
             << " Details:\n"
             << "   Result:\n" << rs << "\n"
             << "   Expected result:\n( 0  0  0  4 )\n( 0 -2  4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c find() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c find() member function of the Rows specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testFind()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::find()";

      initialize();

      using ConstIterator = RT::ConstIterator;

      RT rs = blaze::rows( mat_, { 1UL, 2UL, 3UL } );

      checkRows    ( rs, 3UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 7UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 2UL );
      checkNonZeros( rs, 2UL, 3UL );

      // Searching for the first element
      {
         ConstIterator pos = rs.find( 1UL, 2UL );

         if( pos == rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 2 || pos->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 3\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for the second element
      {
         ConstIterator pos = rs.find( 1UL, 3UL );

         if( pos == rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 4\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for a non-existing non-zero element
      {
         ConstIterator pos = rs.find( 1UL, 1UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Non-existing element could be found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 0\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::find()";

      using ConstIterator = ORT::ConstIterator;

      initialize();

      ORT rs = blaze::rows( tmat_, { 1UL, 2UL, 3UL } );

      checkRows    ( rs, 3UL );
      checkColumns ( rs, 4UL );
      checkNonZeros( rs, 7UL );
      checkNonZeros( rs, 0UL, 2UL );
      checkNonZeros( rs, 1UL, 2UL );
      checkNonZeros( rs, 2UL, 3UL );

      // Searching for the first element
      {
         ConstIterator pos = rs.find( 1UL, 2UL );

         if( pos == rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 2 || pos->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 3\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for the second element
      {
         ConstIterator pos = rs.find( 1UL, 3UL );

         if( pos == rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 4\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for a non-existing non-zero element
      {
         ConstIterator pos = rs.find( 1UL, 1UL );

         if( pos != rs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Non-existing element could be found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 0\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c lowerBound() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c lowerBound() member function of the Rows
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testLowerBound()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::lowerBound()";

      using ConstIterator = RT::ConstIterator;

      initialize();

      RT rs = blaze::rows( mat_, { 1UL } );

      // Determining the lower bound for position (0,0)
      {
         ConstIterator pos = rs.lowerBound( 0UL, 0UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
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
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (0,1)
      {
         ConstIterator pos = rs.lowerBound( 0UL, 1UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Current row selection:\n" << rs << "\n";
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
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (0,2)
      {
         ConstIterator pos = rs.lowerBound( 0UL, 2UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -2\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::lowerBound()";

      using ConstIterator = ORT::ConstIterator;

      initialize();

      ORT rs = blaze::rows( tmat_, { 1UL } );

      // Determining the lower bound for position (0,0)
      {
         ConstIterator pos = rs.lowerBound( 0UL, 0UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
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
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (0,1)
      {
         ConstIterator pos = rs.lowerBound( 0UL, 1UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Current row selection:\n" << rs << "\n";
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
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (0,2)
      {
         ConstIterator pos = rs.lowerBound( 0UL, 2UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -2\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c upperBound() member function of the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c upperBound() member function of the Rows
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testUpperBound()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Rows::upperBound()";

      using ConstIterator = RT::ConstIterator;

      initialize();

      RT rs = blaze::rows( mat_, { 1UL } );

      // Determining the upper bound for position (0,0)
      {
         ConstIterator pos = rs.upperBound( 0UL, 0UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
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
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (0,1)
      {
         ConstIterator pos = rs.upperBound( 0UL, 1UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -2\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (0,2)
      {
         ConstIterator pos = rs.upperBound( 0UL, 2UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -2\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Rows::upperBound()";

      using ConstIterator = ORT::ConstIterator;

      initialize();

      ORT rs = blaze::rows( tmat_, { 1UL } );

      // Determining the upper bound for position (0,0)
      {
         ConstIterator pos = rs.upperBound( 0UL, 0UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
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
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (0,1)
      {
         ConstIterator pos = rs.upperBound( 0UL, 1UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -2\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (0,2)
      {
         ConstIterator pos = rs.upperBound( 0UL, 2UL );

         if( pos == rs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
         else if( pos->index() != 3 || pos->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Wrong element found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = -2\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current row selection:\n" << rs << "\n";
            throw std::runtime_error( oss.str() );
         }
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
void SparseSymmetricTest::testIsDefault()
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
         RT rs = blaze::rows( mat_, { 0UL } );

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
         RT rs = blaze::rows( mat_, { 1UL } );

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
         ORT rs = blaze::rows( tmat_, { 0UL } );

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
         ORT rs = blaze::rows( tmat_, { 1UL } );

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
void SparseSymmetricTest::testIsSame()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function";

      // isSame with matrix and matching row selection
      {
         auto rs = blaze::rows( mat_, { 0UL, 1UL, 2UL, 3UL } );

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
         auto rs = blaze::rows( mat_, { 0UL, 1UL, 2UL } );

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
         auto rs = blaze::rows( mat_, { 0UL, 2UL, 1UL, 3UL } );

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
         auto rs = blaze::rows( mat_, { 0UL, 1UL, 1UL, 3UL } );

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
         auto sm = blaze::submatrix( mat_, 0UL, 0UL, 3UL, 4UL );

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
         auto rs = blaze::rows( tmat_, { 0UL, 1UL, 2UL, 3UL } );

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
         auto rs = blaze::rows( tmat_, { 0UL, 1UL, 2UL } );

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
         auto rs = blaze::rows( tmat_, { 0UL, 2UL, 1UL, 3UL } );

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
         auto rs = blaze::rows( tmat_, { 0UL, 1UL, 1UL, 3UL } );

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
         auto sm = blaze::submatrix( tmat_, 0UL, 0UL, 3UL, 4UL );

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
/*!\brief Test of the \c blaze::submatrix() function with the Rows class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c blaze::submatrix() function with the Rows
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testSubmatrix()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major submatrix() function";

      initialize();

      {
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 2UL );

         if( sm(0,0) !=  0 || sm(0,1) != 3 ||
             sm(1,0) !=  1 || sm(1,1) != 0 ||
             sm(2,0) != -2 || sm(2,1) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0  3 )\n"
                                        "(  1  0 )\n"
                                        "( -2  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm.begin(1UL)->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << sm.begin(1UL)->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 5UL );

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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 2UL );

         if( sm(0,0) !=  0 || sm(0,1) != 3 ||
             sm(1,0) !=  1 || sm(1,1) != 0 ||
             sm(2,0) != -2 || sm(2,1) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n(  0  3 )\n"
                                        "(  1  0 )\n"
                                        "( -2  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( sm.begin(1UL)->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << sm.begin(1UL)->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( rs, 0UL, 1UL, 3UL, 5UL );

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
void SparseSymmetricTest::testRow()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major row() function";

      initialize();

      {
         RT   rs   = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         auto row1 = row( rs, 1UL );

         if( row1[0] != 0 || row1[1] != 1 || row1[2] != 0 || row1[3] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 1 0 -2 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( row1.begin()->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << row1.begin()->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   rs   = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs   = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         auto row1 = row( rs, 1UL );

         if( row1[0] != 0 || row1[1] != 1 || row1[2] != 0 || row1[3] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 1 0 -2 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( row1.begin()->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << row1.begin()->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   rs   = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
void SparseSymmetricTest::testRows()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major rows() function";

      initialize();

      {
         RT rs1 = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         RT rs2 = blaze::rows( rs1, { 1UL, 0UL, 2UL } );

         if( rs2(0,0) != 0 || rs2(0,1) !=  1 || rs2(0,2) != 0 || rs2(0,3) != -2 ||
             rs2(1,0) != 0 || rs2(1,1) !=  0 || rs2(1,2) != 3 || rs2(1,3) !=  4 ||
             rs2(2,0) != 0 || rs2(2,1) != -2 || rs2(2,2) != 4 || rs2(2,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n"
                << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs2.begin( 2UL )->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << rs2.begin( 2UL )->value() << "\n"
                << "   Expected result: -2\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT rs1 = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         RT rs2 = blaze::rows( rs1, { 3UL } );

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
         ORT rs1 = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         ORT rs2 = blaze::rows( rs1, { 1UL, 0UL, 2UL } );

         if( rs2(0,0) != 0 || rs2(0,1) !=  1 || rs2(0,2) != 0 || rs2(0,3) != -2 ||
             rs2(1,0) != 0 || rs2(1,1) !=  0 || rs2(1,2) != 3 || rs2(1,3) !=  4 ||
             rs2(2,0) != 0 || rs2(2,1) != -2 || rs2(2,2) != 4 || rs2(2,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs2 << "\n"
                << "   Expected result:\n( 0  1  0 -2 )\n( 0  0  3  4 )\n( 0 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs2.begin( 2UL )->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << rs2.begin( 2UL )->value() << "\n"
                << "   Expected result: -2\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ORT rs1 = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         ORT rs2 = blaze::rows( rs1, { 3UL } );

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
void SparseSymmetricTest::testColumn()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major column() function";

      initialize();

      {
         RT   rs   = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         auto col1 = blaze::column( rs, 1UL );

         if( col1[0] != 0 || col1[1] != 1 || col1[2] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 0 1 -2 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( col1.begin()->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << col1.begin()->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   rs   = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs   = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         auto col1 = blaze::column( rs, 1UL );

         if( col1[0] != 0 || col1[1] != 1 || col1[2] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 0 1 -2 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( col1.begin()->value() != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << col1.begin()->value() << "\n"
                << "   Expected result: 1\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ORT  rs   = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
void SparseSymmetricTest::testColumns()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major columns() function";

      initialize();

      {
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         auto cs = blaze::columns( rs, { 1UL, 0UL, 2UL } );

         if( cs(0,0) != 0 || cs(0,1) != 0 || cs(0,2) != 3 ||
             cs(1,0) != 1 || cs(1,1) != 0 || cs(1,2) != 0 ||
             cs(2,0) != 2 || cs(2,1) != 0 || cs(2,2) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  3 )\n( 1  0  0 )\n( 2  0  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs.begin( 2UL )->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << cs.begin( 2UL )->value() << "\n"
                << "   Expected result: 3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         auto cs = blaze::columns( rs, { 1UL, 0UL, 2UL } );

         if( cs(0,0) != 0 || cs(0,1) != 0 || cs(0,2) != 3 ||
             cs(1,0) != 1 || cs(1,1) != 0 || cs(1,2) != 0 ||
             cs(2,0) != 2 || cs(2,1) != 0 || cs(2,2) != 4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  3 )\n( 1  0  0 )\n( 2  0  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs.begin( 2UL )->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << cs.begin( 2UL )->value() << "\n"
                << "   Expected result: 3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
void SparseSymmetricTest::testBand()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major band() function";

      initialize();

      {
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
         auto b1 = blaze::band( rs, 1L );

         if( b1[0] != 0 || b1[1] != 0 || b1[2] != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result\n: ( 0 0 5 )\n";
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
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         RT   rs = blaze::rows( mat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
         auto b1 = blaze::band( rs, 1L );

         if( b1[0] != 0 || b1[1] != 0 || b1[2] != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << b1 << "\n"
                << "   Expected result\n: ( 0 0 5 )\n";
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
         ORT  rs = blaze::rows( tmat_, { 2UL, 1UL, 3UL } );
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
void SparseSymmetricTest::initialize()
{
   // Initializing the symmetric row-major matrix
   mat_.reset();
   mat_(1,1) =  1;
   mat_(1,3) = -2;
   mat_(2,2) =  3;
   mat_(2,3) =  4;
   mat_(3,3) =  5;

   // Initializing the symmetric column-major matrix
   tmat_.reset();
   tmat_(1,1) =  1;
   tmat_(1,3) = -2;
   tmat_(2,2) =  3;
   tmat_(2,3) =  4;
   tmat_(3,3) =  5;
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
   std::cout << "   Running Rows sparse symmetric test..." << std::endl;

   try
   {
      RUN_ROWS_SPARSESYMMETRIC_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Rows sparse symmetric test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
