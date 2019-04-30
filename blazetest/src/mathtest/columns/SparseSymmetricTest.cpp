//=================================================================================================
/*!
//  \file src/mathtest/columns/SparseSymmetricTest.cpp
//  \brief Source file for the Columns sparse symmetric test
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
#include <blazetest/mathtest/columns/SparseSymmetricTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace columns {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the Columns sparse symmetric test.
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
/*!\brief Test of the Columns constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the Columns specialization. In case
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
      test_ = "Row-major Columns constructor (index_sequence)";

      initialize();

      // Setup of a regular column selection
      {
         auto cs = blaze::columns( mat_, index_sequence<0,3,2>() );

         if( cs.rows() != mat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != mat_(0,0) || cs(0,1) != mat_(0,3) || cs(0,2) != mat_(0,2) ||
             cs(1,0) != mat_(1,0) || cs(1,1) != mat_(1,3) || cs(1,2) != mat_(1,2) ||
             cs(2,0) != mat_(2,0) || cs(2,1) != mat_(2,3) || cs(2,2) != mat_(2,2) ||
             cs(3,0) != mat_(3,0) || cs(3,1) != mat_(3,3) || cs(3,2) != mat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         auto cs = blaze::columns( mat_, index_sequence<5>() );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( mat_, index_sequence<0,3,2>() );
         auto cs2 = blaze::columns( cs1, index_sequence<2,1>() );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( mat_, { 0, 3, 2 } );
         auto cs2 = blaze::columns( cs1, index_sequence<2,1>() );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs1 = blaze::columns( mat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto cs2 = blaze::columns( cs1, index_sequence<2,1>() );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via initializer_list
   //=====================================================================================

   {
      test_ = "Row-major Columns constructor (initializer_list)";

      initialize();

      // Setup of empty column selection
      {
         std::initializer_list<size_t> indices{};
         auto cs = blaze::columns( mat_, indices );

         if( cs.rows() != mat_.rows() || cs.columns() != 0UL ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular column selection
      {
         auto cs = blaze::columns( mat_, { 0, 3, 2 } );

         if( cs.rows() != mat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != mat_(0,0) || cs(0,1) != mat_(0,3) || cs(0,2) != mat_(0,2) ||
             cs(1,0) != mat_(1,0) || cs(1,1) != mat_(1,3) || cs(1,2) != mat_(1,2) ||
             cs(2,0) != mat_(2,0) || cs(2,1) != mat_(2,3) || cs(2,2) != mat_(2,2) ||
             cs(3,0) != mat_(3,0) || cs(3,1) != mat_(3,3) || cs(3,2) != mat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         auto cs = blaze::columns( mat_, { 5 } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( mat_, index_sequence<0,3,2>() );
         auto cs2 = blaze::columns( cs1, { 2, 1 } );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( mat_, { 0, 3, 2 } );
         auto cs2 = blaze::columns( cs1, { 2, 1 } );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs1 = blaze::columns( mat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto cs2 = blaze::columns( cs1, { 2, 1 } );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via std::vector
   //=====================================================================================

   {
      test_ = "Row-major Columns constructor (std::vector)";

      initialize();

      // Setup of empty column selection
      {
         std::vector<size_t> indices;
         auto cs = blaze::columns( mat_, indices );

         if( cs.rows() != mat_.rows() || cs.columns() != 0UL ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular column selection
      {
         const std::vector<size_t> indices{ 0, 3, 2 };
         auto cs = blaze::columns( mat_, indices );

         if( cs.rows() != mat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != mat_(0,0) || cs(0,1) != mat_(0,3) || cs(0,2) != mat_(0,2) ||
             cs(1,0) != mat_(1,0) || cs(1,1) != mat_(1,3) || cs(1,2) != mat_(1,2) ||
             cs(2,0) != mat_(2,0) || cs(2,1) != mat_(2,3) || cs(2,2) != mat_(2,2) ||
             cs(3,0) != mat_(3,0) || cs(3,1) != mat_(3,3) || cs(3,2) != mat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         std::vector<size_t> indices{ 5 };
         auto cs = blaze::columns( mat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( mat_, index_sequence<0,3,2>() );

         const std::vector<size_t> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( mat_, { 0, 3, 2 } );

         const std::vector<size_t> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto cs1 = blaze::columns( mat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::vector<size_t> indices2{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices2 );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via std::array
   //=====================================================================================

   {
      test_ = "Row-major Columns constructor (std::array)";

      initialize();

      // Setup of a regular column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs = blaze::columns( mat_, indices );

         if( cs.rows() != mat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != mat_(0,0) || cs(0,1) != mat_(0,3) || cs(0,2) != mat_(0,2) ||
             cs(1,0) != mat_(1,0) || cs(1,1) != mat_(1,3) || cs(1,2) != mat_(1,2) ||
             cs(2,0) != mat_(2,0) || cs(2,1) != mat_(2,3) || cs(2,2) != mat_(2,2) ||
             cs(3,0) != mat_(3,0) || cs(3,1) != mat_(3,3) || cs(3,2) != mat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         std::array<size_t,1UL> indices{ 5 };
         auto cs = blaze::columns( mat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( mat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( mat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto cs1 = blaze::columns( mat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup via lambda expression
   //=====================================================================================

   {
      test_ = "Row-major Columns constructor (lambda expression)";

      initialize();

      // Setup of empty column selection
      {
         auto cs = blaze::columns( mat_, []( size_t ){ return 0UL; }, 0UL );

         if( cs.rows() != mat_.rows() || cs.columns() != 0UL ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs = blaze::columns( mat_, [indices]( size_t i ){ return indices[i]; }, 3UL );

         if( cs.rows() != mat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != mat_(0,0) || cs(0,1) != mat_(0,3) || cs(0,2) != mat_(0,2) ||
             cs(1,0) != mat_(1,0) || cs(1,1) != mat_(1,3) || cs(1,2) != mat_(1,2) ||
             cs(2,0) != mat_(2,0) || cs(2,1) != mat_(2,3) || cs(2,2) != mat_(2,2) ||
             cs(3,0) != mat_(3,0) || cs(3,1) != mat_(3,3) || cs(3,2) != mat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         auto cs = blaze::columns( mat_, []( size_t ){ return 5UL; }, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( mat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( mat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto cs1 = blaze::columns( mat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices2{ 2, 1 };
         auto cs2 = blaze::columns( cs1, [indices2]( size_t i ){ return indices2[i]; }, 2UL );

         if( cs2.rows() != mat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != mat_(0,2) || cs2(0,1) != mat_(0,3) ||
             cs2(1,0) != mat_(1,2) || cs2(1,1) != mat_(1,3) ||
             cs2(2,0) != mat_(2,2) || cs2(2,1) != mat_(2,3) ||
             cs2(3,0) != mat_(3,2) || cs2(3,1) != mat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major setup of random in-bounds element selection
   //=====================================================================================

   {
      test_ = "Column-major Columns constructor (stress test)";

      initialize();

      for( size_t rep=0UL; rep<100UL; ++rep )
      {
         blaze::DynamicVector<size_t> indices( blaze::rand<size_t>( 1UL, 20UL ) );
         randomize( indices, 0UL, mat_.rows()-1UL );
         auto cs = blaze::columns( mat_, indices.data(), indices.size() );

         for( size_t i=0UL; i<cs.rows(); ++i ) {
            for( size_t j=0UL; j<cs.columns(); ++j ) {
               if( cs(i,j) != mat_(i,indices[j]) ) {
                  std::ostringstream oss;
                  oss << " Test: " << test_ << "\n"
                      << " Error: Setup of column selection failed\n"
                      << " Details:\n"
                      << "   Indices:\n" << indices << "\n"
                      << "   Column selection:\n" << cs << "\n"
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
      test_ = "Column-major Columns constructor (index_sequence)";

      initialize();

      // Setup of a regular column selection
      {
         auto cs = blaze::columns( tmat_, index_sequence<0,3,2>() );

         if( cs.rows() != tmat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != tmat_(0,0) || cs(0,1) != tmat_(0,3) || cs(0,2) != tmat_(0,2) ||
             cs(1,0) != tmat_(1,0) || cs(1,1) != tmat_(1,3) || cs(1,2) != tmat_(1,2) ||
             cs(2,0) != tmat_(2,0) || cs(2,1) != tmat_(2,3) || cs(2,2) != tmat_(2,2) ||
             cs(3,0) != tmat_(3,0) || cs(3,1) != tmat_(3,3) || cs(3,2) != tmat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         auto cs = blaze::columns( tmat_, index_sequence<5>() );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( tmat_, index_sequence<0,3,2>() );
         auto cs2 = blaze::columns( cs1, index_sequence<2,1>() );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( tmat_, { 0, 3, 2 } );
         auto cs2 = blaze::columns( cs1, index_sequence<2,1>() );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs1 = blaze::columns( tmat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto cs2 = blaze::columns( cs1, index_sequence<2,1>() );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via initializer_list
   //=====================================================================================

   {
      test_ = "Column-major Columns constructor (initializer_list)";

      initialize();

      // Setup of empty column selection
      {
         std::initializer_list<size_t> indices{};
         auto cs = blaze::columns( tmat_, indices );

         if( cs.rows() != tmat_.rows() || cs.columns() != 0UL ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular column selection
      {
         auto cs = blaze::columns( tmat_, { 0, 3, 2 } );

         if( cs.rows() != tmat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != tmat_(0,0) || cs(0,1) != tmat_(0,3) || cs(0,2) != tmat_(0,2) ||
             cs(1,0) != tmat_(1,0) || cs(1,1) != tmat_(1,3) || cs(1,2) != tmat_(1,2) ||
             cs(2,0) != tmat_(2,0) || cs(2,1) != tmat_(2,3) || cs(2,2) != tmat_(2,2) ||
             cs(3,0) != tmat_(3,0) || cs(3,1) != tmat_(3,3) || cs(3,2) != tmat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         auto cs = blaze::columns( tmat_, { 5 } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( tmat_, index_sequence<0,3,2>() );
         auto cs2 = blaze::columns( cs1, { 2, 1 } );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( tmat_, { 0, 3, 2 } );
         auto cs2 = blaze::columns( cs1, { 2, 1 } );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs1 = blaze::columns( tmat_, [indices]( size_t i ){ return indices[i]; }, 3UL );
         auto cs2 = blaze::columns( cs1, { 2, 1 } );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via std::vector
   //=====================================================================================

   {
      test_ = "Column-major Columns constructor (std::vector)";

      initialize();

      // Setup of empty column selection
      {
         std::vector<size_t> indices;
         auto cs = blaze::columns( tmat_, indices );

         if( cs.rows() != tmat_.rows() || cs.columns() != 0UL ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular column selection
      {
         const std::vector<size_t> indices{ 0, 3, 2 };
         auto cs = blaze::columns( tmat_, indices );

         if( cs.rows() != tmat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != tmat_(0,0) || cs(0,1) != tmat_(0,3) || cs(0,2) != tmat_(0,2) ||
             cs(1,0) != tmat_(1,0) || cs(1,1) != tmat_(1,3) || cs(1,2) != tmat_(1,2) ||
             cs(2,0) != tmat_(2,0) || cs(2,1) != tmat_(2,3) || cs(2,2) != tmat_(2,2) ||
             cs(3,0) != tmat_(3,0) || cs(3,1) != tmat_(3,3) || cs(3,2) != tmat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         std::vector<size_t> indices{ 5 };
         auto cs = blaze::columns( tmat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( tmat_, index_sequence<0,3,2>() );

         const std::vector<size_t> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( tmat_, { 0, 3, 2 } );

         const std::vector<size_t> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto cs1 = blaze::columns( tmat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::vector<size_t> indices2{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices2 );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via std::array
   //=====================================================================================

   {
      test_ = "Column-major Columns constructor (std::array)";

      initialize();

      // Setup of a regular column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs = blaze::columns( tmat_, indices );

         if( cs.rows() != tmat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != tmat_(0,0) || cs(0,1) != tmat_(0,3) || cs(0,2) != tmat_(0,2) ||
             cs(1,0) != tmat_(1,0) || cs(1,1) != tmat_(1,3) || cs(1,2) != tmat_(1,2) ||
             cs(2,0) != tmat_(2,0) || cs(2,1) != tmat_(2,3) || cs(2,2) != tmat_(2,2) ||
             cs(3,0) != tmat_(3,0) || cs(3,1) != tmat_(3,3) || cs(3,2) != tmat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         std::array<size_t,1UL> indices{ 5 };
         auto cs = blaze::columns( tmat_, indices );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( tmat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( tmat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto cs1 = blaze::columns( tmat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, indices );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup via lambda expression
   //=====================================================================================

   {
      test_ = "Column-major Columns constructor (lambda expression)";

      initialize();

      // Setup of empty column selection
      {
         auto cs = blaze::columns( tmat_, []( size_t ){ return 0UL; }, 0UL );

         if( cs.rows() != tmat_.rows() || cs.columns() != 0UL ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of empty column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a regular column selection
      {
         const std::array<size_t,3UL> indices{ 0, 3, 2 };
         auto cs = blaze::columns( tmat_, [indices]( size_t i ){ return indices[i]; }, 3UL );

         if( cs.rows() != tmat_.rows() || cs.columns() != 3UL ||
             cs(0,0) != tmat_(0,0) || cs(0,1) != tmat_(0,3) || cs(0,2) != tmat_(0,2) ||
             cs(1,0) != tmat_(1,0) || cs(1,1) != tmat_(1,3) || cs(1,2) != tmat_(1,2) ||
             cs(2,0) != tmat_(2,0) || cs(2,1) != tmat_(2,3) || cs(2,2) != tmat_(2,2) ||
             cs(3,0) != tmat_(3,0) || cs(3,1) != tmat_(3,3) || cs(3,2) != tmat_(3,2) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to setup an out-of-bounds column selection
      try {
         auto cs = blaze::columns( tmat_, []( size_t ){ return 5UL; }, 1UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      // Setup of a column selection on a compile-time column selection
      {
         auto cs1 = blaze::columns( tmat_, index_sequence<0,3,2>() );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an explicit column selection
      {
         auto cs1 = blaze::columns( tmat_, { 0, 3, 2 } );

         const std::array<size_t,2UL> indices{ 2, 1 };
         auto cs2 = blaze::columns( cs1, [indices]( size_t i ){ return indices[i]; }, 2UL );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Setup of a column selection on an implicit column selection
      {
         const std::array<size_t,3UL> indices1{ 0, 3, 2 };
         auto cs1 = blaze::columns( tmat_, [indices1]( size_t i ){ return indices1[i]; }, 3UL );

         const std::array<size_t,2UL> indices2{ 2, 1 };
         auto cs2 = blaze::columns( cs1, [indices2]( size_t i ){ return indices2[i]; }, 2UL );

         if( cs2.rows() != tmat_.rows() || cs2.columns() != 2UL ||
             cs2(0,0) != tmat_(0,2) || cs2(0,1) != tmat_(0,3) ||
             cs2(1,0) != tmat_(1,2) || cs2(1,1) != tmat_(1,3) ||
             cs2(2,0) != tmat_(2,2) || cs2(2,1) != tmat_(2,3) ||
             cs2(3,0) != tmat_(3,2) || cs2(3,1) != tmat_(3,3) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Setup of column selection failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major setup of random in-bounds element selection
   //=====================================================================================

   {
      test_ = "Column-major Columns constructor (stress test)";

      initialize();

      for( size_t rep=0UL; rep<100UL; ++rep )
      {
         blaze::DynamicVector<size_t> indices( blaze::rand<size_t>( 1UL, 20UL ) );
         randomize( indices, 0UL, tmat_.rows()-1UL );
         auto cs = blaze::columns( tmat_, indices.data(), indices.size() );

         for( size_t i=0UL; i<cs.rows(); ++i ) {
            for( size_t j=0UL; j<cs.columns(); ++j ) {
               if( cs(i,j) != tmat_(i,indices[j]) ) {
                  std::ostringstream oss;
                  oss << " Test: " << test_ << "\n"
                      << " Error: Setup of column selection failed\n"
                      << " Details:\n"
                      << "   Indices:\n" << indices << "\n"
                      << "   Column selection:\n" << cs << "\n"
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
/*!\brief Test of the Columns function call operator.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of adding and accessing elements via the function call operator
// of the Columns specialization. In case an error is detected, a \a std::runtime_error exception
// is thrown.
*/
void SparseSymmetricTest::testFunctionCall()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::operator()";

      initialize();

      auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );

      // Assignment to the element (1,1)
      {
         cs(1,1) = 9;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 9UL );
         checkNonZeros( cs  , 0UL, 3UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 3UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 9UL );

         if( cs(0,0) !=  0 || cs(0,1) != 0 || cs(0,2) !=  0 ||
             cs(1,0) !=  1 || cs(1,1) != 9 || cs(1,2) != -2 ||
             cs(2,0) !=  9 || cs(2,1) != 3 || cs(2,2) !=  4 ||
             cs(3,0) != -2 || cs(3,1) != 4 || cs(3,2) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n(  0  0  0 )\n(  1  9 -2 )\n(  9  3  4 )\n( -2  4  5 )\n";
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

      // Assignment to the element (1,2)
      {
         cs(1,2) = 0;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 7UL );
         checkNonZeros( cs  , 0UL, 2UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) != 0 || cs(0,2) != 0 ||
             cs(1,0) != 1 || cs(1,1) != 9 || cs(1,2) != 0 ||
             cs(2,0) != 9 || cs(2,1) != 3 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) != 4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 1  9  0 )\n( 9  3  4 )\n( 0  4  5 )\n";
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

      // Assignment to the element (2,1)
      {
         cs(2,1) = 11;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 7UL );
         checkNonZeros( cs  , 0UL, 2UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 1 || cs(1,1) !=  9 || cs(1,2) != 0 ||
             cs(2,0) != 9 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 1  9  0 )\n( 9 11  4 )\n( 0  4  5 )\n";
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

      // Addition assignment to the element (1,0)
      {
         cs(1,0) += 3;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 7UL );
         checkNonZeros( cs  , 0UL, 2UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  9 || cs(1,2) != 0 ||
             cs(2,0) != 9 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  9  0 )\n( 9 11  4 )\n( 0  4  5 )\n";
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

      // Subtraction assignment to the element (2,0)
      {
         cs(2,0) -= 6;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 7UL );
         checkNonZeros( cs  , 0UL, 2UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  3 || cs(1,2) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  3  0 )\n( 3 11  4 )\n( 0  4  5 )\n";
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

      // Multiplication assignment to the element (2,1)
      {
         cs(2,1) *= 2;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 7UL );
         checkNonZeros( cs  , 0UL, 2UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  3 || cs(1,2) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 22 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  3  0 )\n( 3 22  4 )\n( 0  4  5 )\n";
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

      // Division assignment to the element (2,1)
      {
         cs(2,1) /= 2;

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 3UL );
         checkNonZeros( cs  , 7UL );
         checkNonZeros( cs  , 0UL, 2UL );
         checkNonZeros( cs  , 1UL, 3UL );
         checkNonZeros( cs  , 2UL, 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  3 || cs(1,2) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  3  0 )\n( 3 11  4 )\n( 0  4  5 )\n";
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
      test_ = "Column-major Columns::operator()";

      initialize();

      auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );

      // Assignment to the element (1,1)
      {
         cs(1,1) = 9;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 9UL );
         checkNonZeros( cs   , 0UL, 3UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 3UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 9UL );

         if( cs(0,0) !=  0 || cs(0,1) != 0 || cs(0,2) !=  0 ||
             cs(1,0) !=  1 || cs(1,1) != 9 || cs(1,2) != -2 ||
             cs(2,0) !=  9 || cs(2,1) != 3 || cs(2,2) !=  4 ||
             cs(3,0) != -2 || cs(3,1) != 4 || cs(3,2) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n(  0  0  0 )\n(  1  9 -2 )\n(  9  3  4 )\n( -2  4  5 )\n";
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

      // Assignment to the element (1,2)
      {
         cs(1,2) = 0;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 7UL );
         checkNonZeros( cs   , 0UL, 2UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) != 0 || cs(0,2) != 0 ||
             cs(1,0) != 1 || cs(1,1) != 9 || cs(1,2) != 0 ||
             cs(2,0) != 9 || cs(2,1) != 3 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) != 4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 1  9  0 )\n( 9  3  4 )\n( 0  4  5 )\n";
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

      // Assignment to the element (2,1)
      {
         cs(2,1) = 11;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 7UL );
         checkNonZeros( cs   , 0UL, 2UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 1 || cs(1,1) !=  9 || cs(1,2) != 0 ||
             cs(2,0) != 9 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 1  9  0 )\n( 9 11  4 )\n( 0  4  5 )\n";
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

      // Addition assignment to the element (1,0)
      {
         cs(1,0) += 3;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 7UL );
         checkNonZeros( cs   , 0UL, 2UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  9 || cs(1,2) != 0 ||
             cs(2,0) != 9 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  9  0 )\n( 9 11  4 )\n( 0  4  5 )\n";
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

      // Subtraction assignment to the element (2,0)
      {
         cs(2,0) -= 6;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 7UL );
         checkNonZeros( cs   , 0UL, 2UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  3 || cs(1,2) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  3  0 )\n( 3 11  4 )\n( 0  4  5 )\n";
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

      // Multiplication assignment to the element (2,1)
      {
         cs(2,1) *= 2;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 7UL );
         checkNonZeros( cs   , 0UL, 2UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  3 || cs(1,2) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 22 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  3  0 )\n( 3 22  4 )\n( 0  4  5 )\n";
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

      // Division assignment to the element (2,1)
      {
         cs(2,1) /= 2;

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 3UL );
         checkNonZeros( cs   , 7UL );
         checkNonZeros( cs   , 0UL, 2UL );
         checkNonZeros( cs   , 1UL, 3UL );
         checkNonZeros( cs   , 2UL, 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 7UL );

         if( cs(0,0) != 0 || cs(0,1) !=  0 || cs(0,2) != 0 ||
             cs(1,0) != 4 || cs(1,1) !=  3 || cs(1,2) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 11 || cs(2,2) != 4 ||
             cs(3,0) != 0 || cs(3,1) !=  4 || cs(3,2) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0  0 )\n( 4  3  0 )\n( 3 11  4 )\n( 0  4  5 )\n";
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
/*!\brief Test of the Columns iterator implementation.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the iterator implementation of the Columns specialization.
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

         CT::Iterator it{};

         if( it != CT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Row-major ConstIterator default constructor";

         CT::ConstIterator it{};

         if( it != CT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Row-major Iterator/ConstIterator conversion";

         auto cs = blaze::columns( mat_, { 2UL } );
         auto it( begin( cs, 0UL ) );

         if( it == end( cs, 0UL ) || it->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st column via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         auto cs = blaze::columns( mat_, { 1UL } );
         const ptrdiff_t number( end( cs, 0UL ) - begin( cs, 0UL ) );

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

      // Counting the number of elements in 2nd column via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         auto cs = blaze::columns( mat_, { 2UL } );
         const ptrdiff_t number( cend( cs, 0UL ) - cbegin( cs, 0UL ) );

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

         auto cs = blaze::columns( mat_, { 2UL } );
         auto it ( cbegin( cs, 0UL ) );
         auto end( cend( cs, 0UL ) );

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

         auto cs = blaze::columns( mat_, { 3UL } );
         int value = 6;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it = value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 6 || cs(2,0) != 7 || cs(3,0) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( mat_, { 3UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it += value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 8 || cs(2,0) != 10 || cs(3,0) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( mat_, { 3UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 6 || cs(2,0) != 7 || cs(3,0) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( mat_, { 3UL } );
         int value = 1;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 6 || cs(2,0) != 14 || cs(3,0) != 24 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( mat_, { 3UL } );

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( cs(0,0) != 0 || cs(1,0) != 3 || cs(2,0) != 7 || cs(3,0) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         OCT::Iterator it{};

         if( it != OCT::Iterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing the ConstIterator default constructor
      {
         test_ = "Column-major ConstIterator default constructor";

         OCT::ConstIterator it{};

         if( it != OCT::ConstIterator() ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator default constructor\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing conversion from Iterator to ConstIterator
      {
         test_ = "Column-major Iterator/ConstIterator conversion";

         auto cs = blaze::columns( tmat_, { 2UL } );
         auto it( begin( cs, 0UL ) );

         if( it == end( cs, 0UL ) || it->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st column via Iterator (end-begin)
      {
         test_ = "Column-major Iterator subtraction (end-begin)";

         auto cs = blaze::columns( tmat_, { 1UL } );
         const ptrdiff_t number( end( cs, 0UL ) - begin( cs, 0UL ) );

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

      // Counting the number of elements in 2nd column via ConstIterator (end-begin)
      {
         test_ = "Column-major ConstIterator subtraction (end-begin)";

         auto cs = blaze::columns( tmat_, { 2UL } );
         const ptrdiff_t number( cend( cs, 0UL ) - cbegin( cs, 0UL ) );

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

         auto cs = blaze::columns( tmat_, { 2UL } );
         auto it ( cbegin( cs, 0UL ) );
         auto end( cend( cs, 0UL ) );

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

         auto cs = blaze::columns( tmat_, { 3UL } );
         int value = 6;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it = value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 6 || cs(2,0) != 7 || cs(3,0) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( tmat_, { 3UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it += value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 8 || cs(2,0) != 10 || cs(3,0) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( tmat_, { 3UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 6 || cs(2,0) != 7 || cs(3,0) != 8 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( tmat_, { 3UL } );
         int value = 1;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( cs(0,0) != 0 || cs(1,0) != 6 || cs(2,0) != 14 || cs(3,0) != 24 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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

         auto cs = blaze::columns( tmat_, { 3UL } );

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( cs(0,0) != 0 || cs(1,0) != 3 || cs(2,0) != 7 || cs(3,0) != 12 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
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
/*!\brief Test of the \c nonZeros() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c nonZeros() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testNonZeros()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::nonZeros()";

      initialize();

      // Initialization check
      auto cs = blaze::columns( mat_, { 1UL, 2UL } );

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 2UL );
      checkNonZeros( cs, 4UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 2UL );

      if( cs(0,0) !=  0 || cs(0,1) != 0 ||
          cs(1,0) !=  1 || cs(1,1) != 0 ||
          cs(2,0) !=  0 || cs(2,1) != 3 ||
          cs(3,0) != -2 || cs(3,1) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n(  0 0 )\n(  1 0 )\n(  0 3 )\n( -2 4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the column selection
      cs(2,1) = 0;

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 2UL );
      checkNonZeros( cs, 3UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 1UL );

      if( cs(0,0) !=  0 || cs(0,1) != 0 ||
          cs(1,0) !=  1 || cs(1,1) != 0 ||
          cs(2,0) !=  0 || cs(2,1) != 0 ||
          cs(3,0) != -2 || cs(3,1) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n(  0 0 )\n(  1 0 )\n(  0 0 )\n( -2 4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      mat_(3,2) = 5;

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 2UL );
      checkNonZeros( cs, 3UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 1UL );

      if( cs(0,0) !=  0 || cs(0,1) != 0 ||
          cs(1,0) !=  1 || cs(1,1) != 0 ||
          cs(2,0) !=  0 || cs(2,1) != 0 ||
          cs(3,0) != -2 || cs(3,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n(  0 0 )\n(  1 0 )\n(  0 0 )\n( -2 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::nonZeros()";

      initialize();

      // Initialization check
      auto cs = blaze::columns( tmat_, { 1UL, 2UL } );

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 2UL );
      checkNonZeros( cs, 4UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 2UL );

      if( cs(0,0) !=  0 || cs(0,1) != 0 ||
          cs(1,0) !=  1 || cs(1,1) != 0 ||
          cs(2,0) !=  0 || cs(2,1) != 3 ||
          cs(3,0) != -2 || cs(3,1) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n(  0 0 )\n(  1 0 )\n(  0 3 )\n( -2 4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the column selection
      cs(2,1) = 0;

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 2UL );
      checkNonZeros( cs, 3UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 1UL );

      if( cs(0,0) !=  0 || cs(0,1) != 0 ||
          cs(1,0) !=  1 || cs(1,1) != 0 ||
          cs(2,0) !=  0 || cs(2,1) != 0 ||
          cs(3,0) != -2 || cs(3,1) != 4 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n(  0 0 )\n(  1 0 )\n(  0 0 )\n( -2 4 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Changing the number of non-zeros via the dense matrix
      tmat_(3,2) = 5;

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 2UL );
      checkNonZeros( cs, 3UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 1UL );

      if( cs(0,0) !=  0 || cs(0,1) != 0 ||
          cs(1,0) !=  1 || cs(1,1) != 0 ||
          cs(2,0) !=  0 || cs(2,1) != 0 ||
          cs(3,0) != -2 || cs(3,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Initialization failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n(  0 0 )\n(  1 0 )\n(  0 0 )\n( -2 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c reset() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reset() member function of the Columns specialization.
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

      auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );

      reset( cs(1,0) );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 3UL );
      checkNonZeros( cs  , 6UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( !isDefault( cs(1,0) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0  0 )\n( 0  0 -2 )\n( 0  3  4 )\n( -2  4  5 )\n";
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
      test_ = "Row-major Columns::reset() (lvalue)";

      initialize();

      auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );

      reset( cs );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 3UL );
      checkNonZeros( cs  , 0UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 0UL );

      if( !isDefault( cs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n";
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
      test_ = "Row-major Columns::reset() (rvalue)";

      initialize();

      reset( blaze::columns( mat_, { 1UL, 2UL, 3UL } ) );

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

      auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );

      reset( cs(1,0) );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 3UL );
      checkNonZeros( cs   , 6UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( !isDefault( cs(1,0) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0  0 )\n( 0  0 -2 )\n( 0  3  4 )\n( -2  4  5 )\n";
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
      test_ = "Row-major Columns::reset() (lvalue)";

      initialize();

      auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );

      reset( cs );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 3UL );
      checkNonZeros( cs   , 0UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 0UL );

      if( !isDefault( cs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Reset operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n";
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
      test_ = "Row-major Columns::reset() (rvalue)";

      initialize();

      reset( blaze::columns( tmat_, { 1UL, 2UL, 3UL } ) );

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
/*!\brief Test of the \c clear() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c clear() function with the Columns specialization.
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

      auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );

      clear( cs(1,0) );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 3UL );
      checkNonZeros( cs  , 6UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( !isDefault( cs(1,0) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0  0 )\n( 0  0 -2 )\n( 0  3  4 )\n( -2  4  5 )\n";
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
      test_ = "Row-major Columns::clear() (lvalue)";

      initialize();

      auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );

      clear( cs );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 3UL );
      checkNonZeros( cs  , 0UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 0UL );

      if( !isDefault( cs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n";
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
      test_ = "Row-major Columns::clear() (rvalue)";

      initialize();

      clear( blaze::columns( mat_, { 1UL, 2UL, 3UL } ) );

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

      auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );

      clear( cs(1,0) );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 3UL );
      checkNonZeros( cs   , 6UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( !isDefault( cs(1,0) ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0  0 )\n( 0  0 -2 )\n( 0  3  4 )\n( -2  4  5 )\n";
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
      test_ = "Row-major Columns::clear() (lvalue)";

      initialize();

      auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );

      clear( cs );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 3UL );
      checkNonZeros( cs   , 0UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 0UL );

      if( !isDefault( cs ) ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Clear operation failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n( 0 0 0 )\n";
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
      test_ = "Row-major Columns::clear() (rvalue)";

      initialize();

      clear( blaze::columns( tmat_, { 1UL, 2UL, 3UL } ) );

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
/*!\brief Test of the \c reserve() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c reserve() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testReserve()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::reserve()";

      MT mat( 20UL );

      auto cs = blaze::columns( mat, { 1UL } );

      // Increasing the capacity of the column selection
      cs.reserve( 10UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 10UL );
      checkNonZeros( cs,  0UL );

      // Further increasing the capacity of the column selection
      cs.reserve( 20UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 20UL );
      checkNonZeros( cs,  0UL );
   }

   {
      test_ = "Row-major Columns::reserve( size_t )";

      MT mat( 20UL );

      auto cs = blaze::columns( mat, { 1UL } );

      // Increasing the capacity of a single column
      cs.reserve( 0UL, 10UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 10UL );
      checkNonZeros( cs,  0UL );

      // Further increasing the capacity of a single column
      cs.reserve( 0UL, 15UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 15UL );
      checkNonZeros( cs,  0UL );
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::reserve()";

      OMT mat( 20UL );

      auto cs = blaze::columns( mat, { 1UL } );

      // Increasing the capacity of the column selection
      cs.reserve( 10UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 10UL );
      checkNonZeros( cs,  0UL );

      // Further increasing the capacity of the column selection
      cs.reserve( 20UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 20UL );
      checkNonZeros( cs,  0UL );
   }

   {
      test_ = "Column-major Columns::reserve( size_t )";

      OMT mat( 20UL );

      auto cs = blaze::columns( mat, { 1UL } );

      // Increasing the capacity of a single column
      cs.reserve( 0UL, 10UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 10UL );
      checkNonZeros( cs,  0UL );

      // Further increasing the capacity of a single column
      cs.reserve( 0UL, 15UL );

      checkRows    ( cs, 20UL );
      checkColumns ( cs,  1UL );
      checkCapacity( cs, 15UL );
      checkNonZeros( cs,  0UL );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c trim() member functions of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c trim() member functions of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testTrim()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   // No row-major matrix test required


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::trim()";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Increasing the column capacity of the matrix
      cs.reserve( 0UL, 10UL );
      cs.reserve( 1UL, 20UL );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkCapacity( cs   , 30UL );
      checkCapacity( cs   ,  0UL, 10UL );
      checkCapacity( cs   ,  1UL, 20UL );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, 10UL );
      checkCapacity( tmat_,  3UL, 20UL );

      // Trimming the matrix
      cs.trim();

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkCapacity( cs   , 30UL );
      checkCapacity( cs   ,  0UL, cs.nonZeros( 0UL ) );
      checkCapacity( cs   ,  1UL, cs.nonZeros( 1UL ) );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, tmat_.nonZeros( 2UL ) );
      checkCapacity( tmat_,  3UL, tmat_.nonZeros( 3UL ) );
   }

   {
      test_ = "Column-major Columns::trim( size_t )";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Increasing the column capacity of the matrix
      cs.reserve( 0UL, 10UL );
      cs.reserve( 1UL, 20UL );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkCapacity( cs   , 30UL );
      checkCapacity( cs   ,  0UL, 10UL );
      checkCapacity( cs   ,  1UL, 20UL );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, 10UL );
      checkCapacity( tmat_,  3UL, 20UL );

      // Trimming the 0th column
      cs.trim( 0UL );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkCapacity( cs   , 30UL );
      checkCapacity( cs   ,  0UL, cs.nonZeros( 0UL ) );
      checkCapacity( cs   ,  1UL, 30UL - cs.nonZeros( 0UL ) );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, tmat_.nonZeros( 2UL ) );
      checkCapacity( tmat_,  3UL, 30UL - tmat_.nonZeros( 2UL ) );

      // Trimming the 1st column
      cs.trim( 1UL );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkCapacity( cs   , 30UL );
      checkCapacity( cs   ,  0UL, cs.nonZeros( 0UL ) );
      checkCapacity( cs   ,  1UL, cs.nonZeros( 1UL ) );
      checkCapacity( tmat_, 30UL );
      checkCapacity( tmat_,  2UL, tmat_.nonZeros( 2UL ) );
      checkCapacity( tmat_,  3UL, tmat_.nonZeros( 3UL ) );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c set() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c set() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testSet()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::set()";

      initialize();

      auto cs = blaze::columns( mat_, { 0UL, 1UL } );

      // Setting a non-zero element at the end of the 0th column
      cs.set( 3UL, 0UL, 1 );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the beginning of the 0th column
      cs.set( 0UL, 0UL, 2 );

      checkRows    ( cs  ,  4UL );
      checkColumns ( cs  ,  2UL );
      checkNonZeros( cs  ,  4UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( cs(0,0) != 2 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the center of the 0th column
      cs.set( 1UL, 0UL, 3 );

      checkRows    ( cs  ,  4UL );
      checkColumns ( cs  ,  2UL );
      checkNonZeros( cs  ,  6UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( cs(0,0) != 2 || cs(0,1) !=  3 ||
          cs(1,0) != 3 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting an already existing element
      cs.set( 1UL, 1UL, 4 );

      checkRows    ( cs  ,  4UL );
      checkColumns ( cs  ,  2UL );
      checkNonZeros( cs  ,  6UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( cs(0,0) != 2 || cs(0,1) !=  3 ||
          cs(1,0) != 3 || cs(1,1) !=  4 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  4 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::set()";

      initialize();

      auto cs = blaze::columns( tmat_, { 0UL, 1UL } );

      // Setting a non-zero element at the end of the 0th column
      cs.set( 3UL, 0UL, 1 );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 9UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the beginning of the 0th column
      cs.set( 0UL, 0UL, 2 );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkNonZeros( cs   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( cs(0,0) != 2 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting a non-zero element at the center of the 0th column
      cs.set( 1UL, 0UL, 3 );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkNonZeros( cs   ,  6UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 12UL );

      if( cs(0,0) != 2 || cs(0,1) !=  3 ||
          cs(1,0) != 3 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Setting an already existing element
      cs.set( 1UL, 1UL, 4 );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkNonZeros( cs   ,  6UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 12UL );

      if( cs(0,0) != 2 || cs(0,1) !=  3 ||
          cs(1,0) != 3 || cs(1,1) !=  4 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  4 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c insert() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c insert() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testInsert()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::insert()";

      initialize();

      auto cs = blaze::columns( mat_, { 0UL, 1UL } );

      // Inserting a non-zero element at the end of the 0th column
      cs.insert( 3UL, 0UL, 1 );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 9UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the beginning of the 0th column
      cs.insert( 0UL, 0UL, 2 );

      checkRows    ( cs  ,  4UL );
      checkColumns ( cs  ,  2UL );
      checkNonZeros( cs  ,  4UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 10UL );

      if( cs(0,0) != 2 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the center of the 0th column
      cs.insert( 1UL, 0UL, 3 );

      checkRows    ( cs  ,  4UL );
      checkColumns ( cs  ,  2UL );
      checkNonZeros( cs  ,  6UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 12UL );

      if( cs(0,0) != 2 || cs(0,1) !=  3 ||
          cs(1,0) != 3 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to insert an already existing element
      try {
         cs.insert( 1UL, 1UL, 4 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting an existing element succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  4 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::insert()";

      initialize();

      auto cs = blaze::columns( tmat_, { 0UL, 1UL } );

      // Inserting a non-zero element at the end of the 0th column
      cs.insert( 3UL, 0UL, 1 );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 9UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the beginning of the 0th column
      cs.insert( 0UL, 0UL, 2 );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkNonZeros( cs   ,  4UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 10UL );

      if( cs(0,0) != 2 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  0 )\n( 0  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Inserting a non-zero element at the center of the 0th column
      cs.insert( 1UL, 0UL, 3 );

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkNonZeros( cs   ,  6UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 12UL );

      if( cs(0,0) != 2 || cs(0,1) !=  3 ||
          cs(1,0) != 3 || cs(1,1) !=  1 ||
          cs(2,0) != 0 || cs(2,1) !=  0 ||
          cs(3,0) != 1 || cs(3,1) != -2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  1 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to insert an already existing element
      try {
         cs.insert( 1UL, 1UL, 4 );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Inserting an existing element succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 2  3 )\n( 3  4 )\n( 0  0 )\n( 1 -2 )\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c append() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c append() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testAppend()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::append()";

      // Appending with pre-allocation in each column
      {
         mat_.reset();

         // Initialization check
         auto cs = blaze::columns( mat_, { 3UL, 2UL, 1UL, 0UL } );
         cs.reserve( 0UL, 2UL );
         cs.reserve( 1UL, 1UL );
         cs.reserve( 2UL, 2UL );
         cs.reserve( 3UL, 3UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 0UL );
         checkNonZeros( cs, 0UL, 0UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 0UL );
         checkNonZeros( cs, 3UL, 0UL );

         // Appending one non-zero element
         cs.append( 1UL, 2UL, 1 );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 1UL );
         checkNonZeros( cs, 0UL, 0UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 1UL );
         checkNonZeros( cs, 3UL, 0UL );

         if( cs(1,2) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 1 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.append( 0UL, 0UL, 2 );
         cs.append( 3UL, 0UL, 3 );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 4UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 1UL );
         checkNonZeros( cs, 3UL, 1UL );

         if( cs(1,2) != 1 || cs(0,0) != 2 || cs(3,3) != 2 || cs(3,0) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 2 0 0 0 )\n( 0 0 1 0 )\n( 0 0 0 0 )\n( 3 0 0 2 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.append( 1UL, 3UL, 4 );
         cs.append( 2UL, 3UL, 5 );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 8UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 1UL );
         checkNonZeros( cs, 2UL, 2UL );
         checkNonZeros( cs, 3UL, 3UL );

         if( cs(1,2) != 1 || cs(0,0) != 2 || cs(3,3) != 2 || cs(3,0) != 3 ||
             cs(1,3) != 4 || cs(0,2) != 4 || cs(2,3) != 5 || cs(0,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 2 5 4 0 )\n( 0 0 1 4 )\n( 0 0 0 5 )\n( 3 0 0 2 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Appending with column finalization
      {
         mat_.reset();

         // Initialization check
         auto cs = blaze::columns( mat_, { 3UL, 2UL, 1UL, 0UL } );
         cs.reserve( 0UL, 2UL );
         cs.reserve( 1UL, 2UL );
         cs.reserve( 2UL, 3UL );
         cs.reserve( 3UL, 2UL );

         // Appending one non-zero element
         cs.append( 1UL, 0UL, 1 );
         cs.finalize( 0UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 9UL );
         checkNonZeros( cs, 2UL );
         checkNonZeros( cs, 0UL, 1UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 1UL );
         checkNonZeros( cs, 3UL, 0UL );

         if( cs(1,0) != 1 || cs(3,2) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 1 0 0 0 )\n( 0 0 0 0 )\n( 0 0 1 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.append( 1UL, 1UL, 2 );
         cs.append( 3UL, 1UL, 3 );
         cs.finalize( 1UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 9UL );
         checkNonZeros( cs, 6UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 2UL );
         checkNonZeros( cs, 2UL, 2UL );
         checkNonZeros( cs, 3UL, 0UL );

         if( cs(1,0) != 1 || cs(3,2) != 1 || cs(1,1) != 2 ||
             cs(2,2) != 2 || cs(3,1) != 3 || cs(2,0) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 1 2 0 0 )\n( 3 0 2 0 )\n( 0 3 1 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.finalize( 2UL );
         cs.append( 0UL, 3UL, 4 );
         cs.append( 1UL, 3UL, 5 );
         cs.finalize( 3UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 9UL );
         checkNonZeros( cs, 9UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 2UL );
         checkNonZeros( cs, 2UL, 3UL );
         checkNonZeros( cs, 3UL, 2UL );

         if( cs(1,0) != 1 || cs(3,2) != 1 || cs(1,1) != 2 || cs(2,2) != 2 ||
             cs(3,1) != 3 || cs(2,0) != 3 || cs(0,3) != 4 || cs(0,2) != 5 || cs(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 5 4 )\n( 1 2 0 5 )\n( 3 0 2 0 )\n( 0 3 1 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::append()";

      // Appending with pre-allocation in each column
      {
         tmat_.reset();

         // Initialization check
         auto cs = blaze::columns( tmat_, { 3UL, 2UL, 1UL, 0UL } );
         cs.reserve( 0UL, 2UL );
         cs.reserve( 1UL, 1UL );
         cs.reserve( 2UL, 2UL );
         cs.reserve( 3UL, 3UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 0UL );
         checkNonZeros( cs, 0UL, 0UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 0UL );
         checkNonZeros( cs, 3UL, 0UL );

         // Appending one non-zero element
         cs.append( 1UL, 2UL, 1 );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 1UL );
         checkNonZeros( cs, 0UL, 0UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 1UL );
         checkNonZeros( cs, 3UL, 0UL );

         if( cs(1,2) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 0 0 1 0 )\n( 0 0 0 0 )\n( 0 0 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.append( 0UL, 0UL, 2 );
         cs.append( 3UL, 0UL, 3 );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 4UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 1UL );
         checkNonZeros( cs, 3UL, 1UL );

         if( cs(1,2) != 1 || cs(0,0) != 2 || cs(3,3) != 2 || cs(3,0) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 2 0 0 0 )\n( 0 0 1 0 )\n( 0 0 0 0 )\n( 3 0 0 2 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.append( 1UL, 3UL, 4 );
         cs.append( 2UL, 3UL, 5 );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 8UL );
         checkNonZeros( cs, 8UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 1UL );
         checkNonZeros( cs, 2UL, 2UL );
         checkNonZeros( cs, 3UL, 3UL );

         if( cs(1,2) != 1 || cs(0,0) != 2 || cs(3,3) != 2 || cs(3,0) != 3 ||
             cs(1,3) != 4 || cs(0,2) != 4 || cs(2,3) != 5 || cs(0,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 2 5 4 0 )\n( 0 0 1 4 )\n( 0 0 0 5 )\n( 3 0 0 2 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Appending with column finalization
      {
         tmat_.reset();

         // Initialization check
         auto cs = blaze::columns( tmat_, { 3UL, 2UL, 1UL, 0UL } );
         cs.reserve( 0UL, 2UL );
         cs.reserve( 1UL, 2UL );
         cs.reserve( 2UL, 3UL );
         cs.reserve( 3UL, 2UL );

         // Appending one non-zero element
         cs.append( 1UL, 0UL, 1 );
         cs.finalize( 0UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 9UL );
         checkNonZeros( cs, 2UL );
         checkNonZeros( cs, 0UL, 1UL );
         checkNonZeros( cs, 1UL, 0UL );
         checkNonZeros( cs, 2UL, 1UL );
         checkNonZeros( cs, 3UL, 0UL );

         if( cs(1,0) != 1 || cs(3,2) != 1 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 1 0 0 0 )\n( 0 0 0 0 )\n( 0 0 1 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.append( 1UL, 1UL, 2 );
         cs.append( 3UL, 1UL, 3 );
         cs.finalize( 1UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 9UL );
         checkNonZeros( cs, 6UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 2UL );
         checkNonZeros( cs, 2UL, 2UL );
         checkNonZeros( cs, 3UL, 0UL );

         if( cs(1,0) != 1 || cs(3,2) != 1 || cs(1,1) != 2 ||
             cs(2,2) != 2 || cs(3,1) != 3 || cs(2,0) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 0 0 )\n( 1 2 0 0 )\n( 3 0 2 0 )\n( 0 3 1 0 )\n";
            throw std::runtime_error( oss.str() );
         }

         // Appending two more non-zero elements
         cs.finalize( 2UL );
         cs.append( 0UL, 3UL, 4 );
         cs.append( 1UL, 3UL, 5 );
         cs.finalize( 3UL );

         checkRows    ( cs, 4UL );
         checkColumns ( cs, 4UL );
         checkCapacity( cs, 9UL );
         checkNonZeros( cs, 9UL );
         checkNonZeros( cs, 0UL, 2UL );
         checkNonZeros( cs, 1UL, 2UL );
         checkNonZeros( cs, 2UL, 3UL );
         checkNonZeros( cs, 3UL, 2UL );

         if( cs(1,0) != 1 || cs(3,2) != 1 || cs(1,1) != 2 || cs(2,2) != 2 ||
             cs(3,1) != 3 || cs(2,0) != 3 || cs(0,3) != 4 || cs(0,2) != 5 || cs(1,3) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Append operation failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 5 4 )\n( 1 2 0 5 )\n( 3 0 2 0 )\n( 0 3 1 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c erase() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c erase() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testErase()
{
   //=====================================================================================
   // Row-major index-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Columns::erase( size_t )";

      initialize();

      auto cs = blaze::columns( mat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st column
      cs.erase( 3UL, 1UL );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 4UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 3 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 3  4 )\n( 4  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st column
      cs.erase( 1UL, 1UL );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( cs(0,0) != 0 || cs(0,1) != 0 ||
          cs(1,0) != 0 || cs(1,1) != 0 ||
          cs(2,0) != 3 || cs(2,1) != 4 ||
          cs(3,0) != 4 || cs(3,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 4 )\n( 4 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase an already erased element
      cs.erase( 3UL, 1UL );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 4UL );

      if( cs(0,0) != 0 || cs(0,1) != 0 ||
          cs(1,0) != 0 || cs(1,1) != 0 ||
          cs(2,0) != 3 || cs(2,1) != 4 ||
          cs(3,0) != 4 || cs(3,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 4 )\n( 4 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major iterator-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Columns::erase( size_t, Iterator )";

      initialize();

      auto cs = blaze::columns( mat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 3UL, 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 4UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 6UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) !=  0 ||
             cs(1,0) != 0 || cs(1,1) != -2 ||
             cs(2,0) != 3 || cs(2,1) !=  4 ||
             cs(3,0) != 4 || cs(3,1) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 3  4 )\n( 4  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 1UL, 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 3UL );
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

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 4 ||
             cs(3,0) != 4 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 4 )\n( 4 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 2UL, 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 1UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 2UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an already erased element
      {
         auto pos = cs.erase( 1UL, cs.find( 3UL, 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 1UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 2UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major iterator-range-based erase function
   //=====================================================================================

   {
      test_ = "Row-major Columns::erase( size_t, Iterator, Iterator )";

      initialize();

      auto cs = blaze::columns( mat_, { 2UL, 3UL } );

      // Erasing the 0th column
      {
         auto pos = cs.erase( 0UL, cs.begin( 0UL ), cs.end( 0UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 2UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 4UL );

         if( pos != cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) !=  0 ||
             cs(1,0) != 0 || cs(1,1) != -2 ||
             cs(2,0) != 0 || cs(2,1) !=  0 ||
             cs(3,0) != 0 || cs(3,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  0 )\n( 0  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the first half of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.begin( 1UL ), cs.find( 3UL, 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 1UL );
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

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 0 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the first half of the 1st column failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the second half of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 3UL, 1UL ), cs.end( 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 0UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 1UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 0 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the second half of the 1st column failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an empty range
      {
         auto pos = cs.erase( 1UL, cs.begin( 1UL ), cs.begin( 1UL ) );

         checkRows    ( cs  , 4UL );
         checkColumns ( cs  , 2UL );
         checkNonZeros( cs  , 0UL );
         checkRows    ( mat_, 4UL );
         checkColumns ( mat_, 4UL );
         checkNonZeros( mat_, 1UL );

         if( pos != cs.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the given end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 0 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing an empty range failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major erase function with predicate
   //=====================================================================================

   {
      test_ = "Row-major Columns::erase( Predicate )";

      initialize();

      auto cs = blaze::columns( mat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      cs.erase( []( int value ) { return value == 3 || value == 5; } );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 5UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase all elements with value 1
      cs.erase( []( int value ){ return value == 1; } );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 3UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 5UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing all elements with value 1 failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major iterator-range-based erase function with predicate
   //=====================================================================================

   {
      test_ = "Row-major Columns::erase( size_t, Iterator, Iterator, Predicate )";

      initialize();

      auto cs = blaze::columns( mat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      cs.erase( 0UL, cs.begin( 0UL ), cs.find( 3UL, 0UL ),
                []( int value ) { return value == 3; } );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 4UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase from an empty range
      cs.erase( 1UL, cs.begin( 1UL ), cs.begin( 1UL ), []( int ){ return true; } );

      checkRows    ( cs  , 4UL );
      checkColumns ( cs  , 2UL );
      checkNonZeros( cs  , 4UL );
      checkRows    ( mat_, 4UL );
      checkColumns ( mat_, 4UL );
      checkNonZeros( mat_, 6UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing from an empty range failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major index-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Columns::erase( size_t )";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st column
      cs.erase( 3UL, 1UL );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 3 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 3  4 )\n( 4  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Erasing the non-zero element at the beginning of the 1st column
      cs.erase( 1UL, 1UL );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( cs(0,0) != 0 || cs(0,1) != 0 ||
          cs(1,0) != 0 || cs(1,1) != 0 ||
          cs(2,0) != 3 || cs(2,1) != 4 ||
          cs(3,0) != 4 || cs(3,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 4 )\n( 4 0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase an already erased element
      cs.erase( 3UL, 1UL );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 4UL );

      if( cs(0,0) != 0 || cs(0,1) != 0 ||
          cs(1,0) != 0 || cs(1,1) != 0 ||
          cs(2,0) != 3 || cs(2,1) != 4 ||
          cs(3,0) != 4 || cs(3,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a non-zero element failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 4 )\n( 4 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major iterator-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Columns::erase( size_t, Iterator )";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Erasing the non-zero element at the end of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 3UL, 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 4UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 6UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) !=  0 ||
             cs(1,0) != 0 || cs(1,1) != -2 ||
             cs(2,0) != 3 || cs(2,1) !=  4 ||
             cs(3,0) != 4 || cs(3,1) !=  0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 3  4 )\n( 4  0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 1UL, 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 3UL );
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

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 4 ||
             cs(3,0) != 4 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 4 )\n( 4 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the non-zero element at the beginning of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 2UL, 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 1UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 2UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a non-zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an already erased element
      {
         auto pos = cs.erase( 1UL, cs.find( 3UL, 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 1UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 2UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 3 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing a zero element failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 3 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major iterator-range-based erase function
   //=====================================================================================

   {
      test_ = "Column-major Columns::erase( size_t, Iterator, Iterator )";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Erasing the 0th column
      {
         auto pos = cs.erase( 0UL, cs.begin( 0UL ), cs.end( 0UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 2UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 4UL );

         if( pos != cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) !=  0 ||
             cs(1,0) != 0 || cs(1,1) != -2 ||
             cs(2,0) != 0 || cs(2,1) !=  0 ||
             cs(3,0) != 0 || cs(3,1) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the 0th column failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  0 )\n( 0  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the first half of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.begin( 1UL ), cs.find( 3UL, 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 1UL );
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

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 0 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the first half of the 1st column failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Erasing the second half of the 1st column
      {
         auto pos = cs.erase( 1UL, cs.find( 3UL, 1UL ), cs.end( 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 1UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 0 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing the second half of the 1st column failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Trying to erase an empty range
      {
         auto pos = cs.erase( 1UL, cs.begin( 1UL ), cs.begin( 1UL ) );

         checkRows    ( cs   , 4UL );
         checkColumns ( cs   , 2UL );
         checkNonZeros( cs   , 0UL );
         checkRows    ( tmat_, 4UL );
         checkColumns ( tmat_, 4UL );
         checkNonZeros( tmat_, 1UL );

         if( pos != cs.begin( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid iterator returned\n"
                << " Details:\n"
                << "   Expected result: the given end() iterator\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs(0,0) != 0 || cs(0,1) != 0 ||
             cs(1,0) != 0 || cs(1,1) != 0 ||
             cs(2,0) != 0 || cs(2,1) != 0 ||
             cs(3,0) != 0 || cs(3,1) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Erasing an empty range failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 0 0 )\n( 0 0 )\n( 0 0 )\n( 0 0 )\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major erase function with predicate
   //=====================================================================================

   {
      test_ = "Column-major Columns::erase( Predicate )";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      cs.erase( []( int value ) { return value == 3 || value == 5; } );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 5UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  0 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase all elements with value 1
      cs.erase( []( int value ){ return value == 1; } );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 3UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 5UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing all elements with value 1 failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major iterator-range-based erase function with predicate
   //=====================================================================================

   {
      test_ = "Column-major Columns::erase( size_t, Iterator, Iterator, Predicate )";

      initialize();

      auto cs = blaze::columns( tmat_, { 2UL, 3UL } );

      // Erasing a selection of elements
      cs.erase( 0UL, cs.begin( 0UL ), cs.find( 3UL, 0UL ),
                []( int value ) { return value == 3; } );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing a selection of elements failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  5 )\n";
         throw std::runtime_error( oss.str() );
      }

      // Trying to erase from an empty range
      cs.erase( 1UL, cs.begin( 1UL ), cs.begin( 1UL ), []( int ){ return true; } );

      checkRows    ( cs   , 4UL );
      checkColumns ( cs   , 2UL );
      checkNonZeros( cs   , 4UL );
      checkRows    ( tmat_, 4UL );
      checkColumns ( tmat_, 4UL );
      checkNonZeros( tmat_, 6UL );

      if( cs(0,0) != 0 || cs(0,1) !=  0 ||
          cs(1,0) != 0 || cs(1,1) != -2 ||
          cs(2,0) != 0 || cs(2,1) !=  4 ||
          cs(3,0) != 4 || cs(3,1) !=  5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Erasing from an empty range failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 0  0 )\n( 0 -2 )\n( 0  4 )\n( 4  5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c find() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c find() member function of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testFind()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::find()";

      initialize();

      auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 3UL );
      checkNonZeros( cs, 7UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 2UL );
      checkNonZeros( cs, 2UL, 3UL );

      // Searching for the first element
      {
         auto pos = cs.find( 2UL, 1UL );

         if( pos == cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for the second element
      {
         auto pos = cs.find( 3UL, 1UL );

         if( pos == cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for a non-existing non-zero element
      {
         auto pos = cs.find( 1UL, 1UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Non-existing element could be found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 0\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::find()";

      initialize();

      auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );

      checkRows    ( cs, 4UL );
      checkColumns ( cs, 3UL );
      checkNonZeros( cs, 7UL );
      checkNonZeros( cs, 0UL, 2UL );
      checkNonZeros( cs, 1UL, 2UL );
      checkNonZeros( cs, 2UL, 3UL );

      // Searching for the first element
      {
         auto pos = cs.find( 2UL, 1UL );

         if( pos == cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for the second element
      {
         auto pos = cs.find( 3UL, 1UL );

         if( pos == cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Element could not be found\n"
                << " Details:\n"
                << "   Required index = 3\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Searching for a non-existing non-zero element
      {
         auto pos = cs.find( 1UL, 1UL );

         if( pos != cs.end( 1UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Non-existing element could be found\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Found index    = " << pos->index() << "\n"
                << "   Expected value = 0\n"
                << "   Value at index = " << pos->value() << "\n"
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c lowerBound() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c lowerBound() member function of the Columns
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testLowerBound()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::lowerBound()";

      initialize();

      auto cs = blaze::columns( mat_, { 1UL } );

      // Determining the lower bound for position (0,0)
      {
         auto pos = cs.lowerBound( 0UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (1,0)
      {
         auto pos = cs.lowerBound( 1UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (2,0)
      {
         auto pos = cs.lowerBound( 2UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::lowerBound()";

      initialize();

      auto cs = blaze::columns( tmat_, { 1UL } );

      // Determining the lower bound for position (0,0)
      {
         auto pos = cs.lowerBound( 0UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (1,0)
      {
         auto pos = cs.lowerBound( 1UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 1\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the lower bound for position (2,0)
      {
         auto pos = cs.lowerBound( 2UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Lower bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 2\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c upperBound() member function of the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c upperBound() member function of the Columns
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testUpperBound()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major Columns::upperBound()";

      initialize();

      auto cs = blaze::columns( mat_, { 1UL } );

      // Determining the upper bound for position (0,0)
      {
         auto pos = cs.upperBound( 0UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (1,0)
      {
         auto pos = cs.upperBound( 1UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (2,0)
      {
         auto pos = cs.upperBound( 2UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major Columns::upperBound()";

      initialize();

      auto cs = blaze::columns( tmat_, { 1UL } );

      // Determining the upper bound for position (0,0)
      {
         auto pos = cs.upperBound( 0UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (1,0)
      {
         auto pos = cs.upperBound( 1UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Determining the upper bound for position (2,0)
      {
         auto pos = cs.upperBound( 2UL, 0UL );

         if( pos == cs.end( 0UL ) ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Upper bound could not be determined\n"
                << " Details:\n"
                << "   Required index = 0\n"
                << "   Current column selection:\n" << cs << "\n";
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
                << "   Current column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isDefault() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the Columns specialization.
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

      // isDefault with default column selection
      {
         auto cs = blaze::columns( mat_, { 0UL } );

         if( isDefault( cs(1,0) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column element: " << cs(1,0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( cs ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default column selection
      {
         auto cs = blaze::columns( mat_, { 1UL } );

         if( isDefault( cs(1,0) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column element: " << cs(1,0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( cs ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n";
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

      // isDefault with default column selection
      {
         auto cs = blaze::columns( tmat_, { 0UL } );

         if( isDefault( cs(1,0) ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column element: " << cs(1,0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( cs ) != true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isDefault with non-default column selection
      {
         auto cs = blaze::columns( tmat_, { 1UL } );

         if( isDefault( cs(1,0) ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column element: " << cs(1,0) << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( isDefault( cs ) != false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isDefault evaluation\n"
                << " Details:\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c isSame() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isSame() function with the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void SparseSymmetricTest::testIsSame()
{
   //=====================================================================================
   // Row-major matrix tests
   //=====================================================================================

   {
      test_ = "Row-major isSame() function";

      // isSame with matrix and matching column selection
      {
         auto cs = blaze::columns( mat_, { 0UL, 1UL, 2UL, 3UL } );

         if( blaze::isSame( cs, mat_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching column selection (different number of columns)
      {
         auto cs = blaze::columns( mat_, { 0UL, 1UL, 2UL } );

         if( blaze::isSame( cs, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching column selection (different order of columns)
      {
         auto cs = blaze::columns( mat_, { 0UL, 2UL, 1UL, 3UL } );

         if( blaze::isSame( cs, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching column selection (repeating columns)
      {
         auto cs = blaze::columns( mat_, { 0UL, 1UL, 1UL, 3UL } );

         if( blaze::isSame( cs, mat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( mat_, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << mat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and matching column selection
      {
         auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 0UL, 1UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different number of rows)
      {
         auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 0UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different number of columns)
      {
         auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 0UL, 1UL, 4UL, 2UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different order of columns)
      {
         auto cs = blaze::columns( mat_, { 1UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( mat_, 0UL, 1UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (repeating columns)
      {
         auto cs = blaze::columns( mat_, { 1UL, 3UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 0UL, 1UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different column index)
      {
         auto cs = blaze::columns( mat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( mat_, 0UL, 0UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching column selections
      {
         auto cs1 = blaze::columns( mat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( mat_, { 0UL, 3UL, 1UL } );

         if( blaze::isSame( cs1, cs2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching column selections (different number of columns)
      {
         auto cs1 = blaze::columns( mat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( mat_, { 0UL, 3UL, 1UL, 2UL } );

         if( blaze::isSame( cs1, cs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching column selections (different order of columns)
      {
         auto cs1 = blaze::columns( mat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( mat_, { 0UL, 1UL, 3UL } );

         if( blaze::isSame( cs1, cs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching column selections (repeating columns)
      {
         auto cs1 = blaze::columns( mat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( mat_, { 0UL, 1UL, 1UL } );

         if( blaze::isSame( cs1, cs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Column-major matrix tests
   //=====================================================================================

   {
      test_ = "Column-major isSame() function";

      // isSame with matrix and matching column selection
      {
         auto cs = blaze::columns( tmat_, { 0UL, 1UL, 2UL, 3UL } );

         if( blaze::isSame( cs, tmat_ ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching column selection (different number of columns)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 1UL, 2UL } );

         if( blaze::isSame( cs, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching column selection (different order of columns)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 2UL, 1UL, 3UL } );

         if( blaze::isSame( cs, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matrix and non-matching column selection (repeating columns)
      {
         auto cs = blaze::columns( tmat_, { 0UL, 1UL, 1UL, 3UL } );

         if( blaze::isSame( cs, tmat_ ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( tmat_, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Matrix:\n" << tmat_ << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and matching column selection
      {
         auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different number of rows)
      {
         auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 0UL, 1UL, 3UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different number of columns)
      {
         auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 2UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different order of columns)
      {
         auto cs = blaze::columns( tmat_, { 1UL, 3UL, 2UL } );
         auto sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (repeating columns)
      {
         auto cs = blaze::columns( tmat_, { 1UL, 3UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 0UL, 1UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with submatrix and non-matching column selection (different column index)
      {
         auto cs = blaze::columns( tmat_, { 1UL, 2UL, 3UL } );
         auto sm = blaze::submatrix( tmat_, 0UL, 0UL, 4UL, 3UL );

         if( blaze::isSame( cs, sm ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }

         if( blaze::isSame( sm, cs ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   Submatrix:\n" << sm << "\n"
                << "   Column selection:\n" << cs << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with matching column selections
      {
         auto cs1 = blaze::columns( tmat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( tmat_, { 0UL, 3UL, 1UL } );

         if( blaze::isSame( cs1, cs2 ) == false ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching column selections (different number of columns)
      {
         auto cs1 = blaze::columns( tmat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( tmat_, { 0UL, 3UL, 1UL, 2UL } );

         if( blaze::isSame( cs1, cs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching column selections (different order of columns)
      {
         auto cs1 = blaze::columns( tmat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( tmat_, { 0UL, 1UL, 3UL } );

         if( blaze::isSame( cs1, cs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // isSame with non-matching column selections (repeating columns)
      {
         auto cs1 = blaze::columns( tmat_, { 0UL, 3UL, 1UL } );
         auto cs2 = blaze::columns( tmat_, { 0UL, 1UL, 1UL } );

         if( blaze::isSame( cs1, cs2 ) == true ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid isSame evaluation\n"
                << " Details:\n"
                << "   First column selection:\n" << cs1 << "\n"
                << "   Second column selection:\n" << cs2 << "\n";
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
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 2UL, 3UL );

         if( sm(0,0) != 0 || sm(0,1) != 1 || sm(0,2) != -2 ||
             sm(1,0) != 3 || sm(1,1) != 0 || sm(1,2) !=  4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0  1 -2 )\n"
                                        "( 3  0  4 )\n";
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
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 4UL, 0UL, 2UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 3UL, 2UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 5UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 2UL, 4UL );

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
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 2UL, 3UL );

         if( sm(0,0) != 0 || sm(0,1) != 1 || sm(0,2) != -2 ||
             sm(1,0) != 3 || sm(1,1) != 0 || sm(1,2) !=  4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << sm << "\n"
                << "   Expected result:\n( 0  1 -2 )\n"
                                        "( 3  0  4 )\n";
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
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 4UL, 0UL, 2UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 3UL, 2UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 5UL, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds submatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << sm << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto sm = blaze::submatrix( cs, 1UL, 0UL, 2UL, 4UL );

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
/*!\brief Test of the \c row() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c row() function with the Columns specialization.
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
         auto cs   = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto row1 = row( cs, 1UL );

         if( row1[0] != 0 || row1[1] != 1 || row1[2] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 1 -2 )\n";
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
         auto cs   = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto row4 = row( cs, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row4 << "\n";
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
         auto cs   = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto row1 = row( cs, 1UL );

         if( row1[0] != 0 || row1[1] != 1 || row1[2] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << row1 << "\n"
                << "   Expected result:\n( 0 1 -2 )\n";
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
         auto cs   = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto row4 = row( cs, 4UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds row succeeded\n"
             << " Details:\n"
             << "   Result:\n" << row4 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c rows() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c rows() function with the Columns specialization.
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
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto rs = blaze::rows( cs, { 1UL, 0UL, 2UL } );

         if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != -2 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) !=  0 ||
             rs(2,0) != 3 || rs(2,1) != 0 || rs(2,2) !=  4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1 -2 )\n( 0  0  0 )\n( 3  0  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs.begin( 2UL )->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << rs.begin( 2UL )->value() << "\n"
                << "   Expected result: 3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto rs = blaze::rows( cs, { 4UL } );

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
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto rs = blaze::rows( cs, { 1UL, 0UL, 2UL } );

         if( rs(0,0) != 0 || rs(0,1) != 1 || rs(0,2) != -2 ||
             rs(1,0) != 0 || rs(1,1) != 0 || rs(1,2) !=  0 ||
             rs(2,0) != 3 || rs(2,1) != 0 || rs(2,2) !=  4 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << rs << "\n"
                << "   Expected result:\n( 0  1 -2 )\n( 0  0  0 )\n( 3  0  4 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( rs.begin( 2UL )->value() != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << rs.begin( 2UL )->value() << "\n"
                << "   Expected result: 3\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto rs = blaze::rows( cs, { 4UL } );

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
/*!\brief Test of the \c column() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c column() function with the Columns specialization.
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
         auto cs   = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto col1 = blaze::column( cs, 1UL );

         if( col1[0] != 0 || col1[1] != 1 || col1[2] != 0 || col1[3] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 0 1 0 -2 )\n";
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
         auto cs   = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto col3 = blaze::column( cs, 3UL );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column succeeded\n"
             << " Details:\n"
             << "   Result:\n" << col3 << "\n";
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
         auto cs   = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto col1 = blaze::column( cs, 1UL );

         if( col1[0] != 0 || col1[1] != 1 || col1[2] != 0 || col1[3] != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Subscript operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << col1 << "\n"
                << "   Expected result:\n( 0 1 0 -2 )\n";
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
         auto cs   = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto col3 = blaze::column( cs, 3UL );

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
/*!\brief Test of the \c columns() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c columns() function with the Columns specialization.
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
         auto cs1 = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto cs2 = blaze::columns( cs1, { 1UL, 0UL, 2UL } );

         if( cs2(0,0) !=  0 || cs2(0,1) != 0 || cs2(0,2) !=  0 ||
             cs2(1,0) !=  1 || cs2(1,1) != 0 || cs2(1,2) != -2 ||
             cs2(2,0) !=  0 || cs2(2,1) != 3 || cs2(2,2) !=  4 ||
             cs2(3,0) != -2 || cs2(3,1) != 4 || cs2(3,2) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n"
                << "   Expected result:\n(  0  0  0 )\n(  1  0 -2 )\n(   0  3  4 )\n( -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs2.begin( 2UL )->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << cs2.begin( 2UL )->value() << "\n"
                << "   Expected result: -2\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto cs1 = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto cs2 = blaze::columns( cs1, { 3UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs2 << "\n";
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
         auto cs1 = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto cs2 = blaze::columns( cs1, { 1UL, 0UL, 2UL } );

         if( cs2(0,0) !=  0 || cs2(0,1) != 0 || cs2(0,2) !=  0 ||
             cs2(1,0) !=  1 || cs2(1,1) != 0 || cs2(1,2) != -2 ||
             cs2(2,0) !=  0 || cs2(2,1) != 3 || cs2(2,2) !=  4 ||
             cs2(3,0) != -2 || cs2(3,1) != 4 || cs2(3,2) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Function call operator access failed\n"
                << " Details:\n"
                << "   Result:\n" << cs2 << "\n"
                << "   Expected result:\n(  0  0  0 )\n(  1  0 -2 )\n(   0  3  4 )\n( -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( cs2.begin( 2UL )->value() != -2 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << cs2.begin( 2UL )->value() << "\n"
                << "   Expected result: -2\n";
            throw std::runtime_error( oss.str() );
         }
      }

      try {
         auto cs1 = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto cs2 = blaze::columns( cs1, { 3UL } );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds column selection succeeded\n"
             << " Details:\n"
             << "   Result:\n" << cs2 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the \c band() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c band() function with the Columns specialization.
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
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto b1 = blaze::band( cs, -1L );

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
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto b3 = blaze::band( cs, 3L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( mat_, { 2UL, 1UL, 3UL } );
         auto b4 = blaze::band( cs, -4L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b4 << "\n";
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
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto b1 = blaze::band( cs, -1L );

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
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto b3 = blaze::band( cs, 3L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b3 << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}

      try {
         auto cs = blaze::columns( tmat_, { 2UL, 1UL, 3UL } );
         auto b4 = blaze::band( cs, -4L );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of out-of-bounds band succeeded\n"
             << " Details:\n"
             << "   Result:\n" << b4 << "\n";
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

} // namespace columns

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
   std::cout << "   Running Columns sparse symmetric test..." << std::endl;

   try
   {
      RUN_COLUMNS_SPARSESYMMETRIC_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Columns sparse symmetric test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
