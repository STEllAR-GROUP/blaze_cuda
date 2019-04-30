//=================================================================================================
/*!
//  \file src/mathtest/columns/DenseSymmetricTest.cpp
//  \brief Source file for the Columns dense symmetric test
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
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Views.h>
#include <blazetest/mathtest/columns/DenseSymmetricTest.h>

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
/*!\brief Constructor for the Columns dense symmetric test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseSymmetricTest::DenseSymmetricTest()
   : mat_ ( 4UL )
   , tmat_( 4UL )
{
   testConstructors();
   testAssignment();
   testFunctionCall();
   testIterator();
   testNonZeros();
   testReset();
   testClear();
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
/*!\brief Test of the Columns constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the Columns specialization. In case
// an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseSymmetricTest::testConstructors()
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
/*!\brief Test of the Columns assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseSymmetricTest::testAssignment()
{
   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Row-major Columns homogeneous assignment";

      initialize();

      auto cs = blaze::columns( mat_, { 3UL, 1UL } );
      cs = 12;

      checkRows    ( cs  ,  4UL );
      checkColumns ( cs  ,  2UL );
      checkNonZeros( cs  ,  8UL );
      checkRows    ( mat_,  4UL );
      checkColumns ( mat_,  4UL );
      checkNonZeros( mat_, 13UL );

      if( cs(0,0) != 12 || cs(0,1) != 12 ||
          cs(1,0) != 12 || cs(1,1) != 12 ||
          cs(2,0) != 12 || cs(2,1) != 12 ||
          cs(3,0) != 12 || cs(3,1) != 12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 12 12 )\n( 12 12 )\n( 12 12 )\n( 12 12 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( mat_(0,0) !=  0 || mat_(0,1) != 12 || mat_(0,2) !=  0 || mat_(0,3) != 12 ||
          mat_(1,0) != 12 || mat_(1,1) != 12 || mat_(1,2) != 12 || mat_(1,3) != 12 ||
          mat_(2,0) !=  0 || mat_(2,1) != 12 || mat_(2,2) !=  3 || mat_(2,3) != 12 ||
          mat_(3,0) != 12 || mat_(3,1) != 12 || mat_(3,2) != 12 || mat_(3,3) != 12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << mat_ << "\n"
             << "   Expected result:\n(  0 12  0 12 )\n"
                                     "( 12 12 12 12 )\n"
                                     "(  0 12  3 12 )\n"
                                     "( 12 12 12 12 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major homogeneous assignment
   //=====================================================================================

   {
      test_ = "Column-major Columns homogeneous assignment";

      initialize();

      auto cs = blaze::columns( tmat_, { 3UL, 1UL } );
      cs = 12;

      checkRows    ( cs   ,  4UL );
      checkColumns ( cs   ,  2UL );
      checkNonZeros( cs   ,  8UL );
      checkRows    ( tmat_,  4UL );
      checkColumns ( tmat_,  4UL );
      checkNonZeros( tmat_, 13UL );

      if( cs(0,0) != 12 || cs(0,1) != 12 ||
          cs(1,0) != 12 || cs(1,1) != 12 ||
          cs(2,0) != 12 || cs(2,1) != 12 ||
          cs(3,0) != 12 || cs(3,1) != 12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << cs << "\n"
             << "   Expected result:\n( 12 12 )\n( 12 12 )\n( 12 12 )\n( 12 12 )\n";
         throw std::runtime_error( oss.str() );
      }

      if( tmat_(0,0) !=  0 || tmat_(0,1) != 12 || tmat_(0,2) !=  0 || tmat_(0,3) != 12 ||
          tmat_(1,0) != 12 || tmat_(1,1) != 12 || tmat_(1,2) != 12 || tmat_(1,3) != 12 ||
          tmat_(2,0) !=  0 || tmat_(2,1) != 12 || tmat_(2,2) !=  3 || tmat_(2,3) != 12 ||
          tmat_(3,0) != 12 || tmat_(3,1) != 12 || tmat_(3,2) != 12 || tmat_(3,3) != 12 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << tmat_ << "\n"
             << "   Expected result:\n(  0 12  0 12 )\n"
                                     "( 12 12 12 12 )\n"
                                     "(  0 12  3 12 )\n"
                                     "( 12 12 12 12 )\n";
         throw std::runtime_error( oss.str() );
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
void DenseSymmetricTest::testFunctionCall()
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
void DenseSymmetricTest::testIterator()
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

         if( it == end( cs, 0UL ) || *it != 0 ) {
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

      // Counting the number of elements in 1st column via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         auto cs = blaze::columns( mat_, { 1UL } );
         const ptrdiff_t number( begin( cs, 0UL ) - end( cs, 0UL ) );

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

      // Counting the number of elements in 2nd column via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         auto cs = blaze::columns( mat_, { 2UL } );
         const ptrdiff_t number( cend( cs, 0UL ) - cbegin( cs, 0UL ) );

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

      // Counting the number of elements in 2nd column via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         auto cs = blaze::columns( mat_, { 2UL } );
         const ptrdiff_t number( cbegin( cs, 0UL ) - cend( cs, 0UL ) );

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

         auto cs = blaze::columns( mat_, { 3UL } );
         auto it ( cbegin( cs, 0UL ) );
         auto end( cend( cs, 0UL ) );

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != -2 ) {
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

         if( it == end || *it != -2 ) {
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

         if( it == end || *it != 4 ) {
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

         if( it == end || *it != 5 ) {
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

         auto cs = blaze::columns( mat_, { 0UL } );
         int value = 6;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it = value++;
         }

         if( cs(0,0) != 6 || cs(1,0) != 7 || cs(2,0) != 8 || cs(3,0) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 6 || mat_(0,1) !=  7 || mat_(0,2) != 8 || mat_(0,3) !=  9 ||
             mat_(1,0) != 7 || mat_(1,1) !=  1 || mat_(1,2) != 0 || mat_(1,3) != -2 ||
             mat_(2,0) != 8 || mat_(2,1) !=  0 || mat_(2,2) != 3 || mat_(2,3) !=  4 ||
             mat_(3,0) != 9 || mat_(3,1) != -2 || mat_(3,2) != 4 || mat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 6  7  8  9 )\n"
                                        "( 7  1  0 -2 )\n"
                                        "( 8  0  3  4 )\n"
                                        "( 9 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         auto cs = blaze::columns( mat_, { 0UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it += value++;
         }

         if( cs(0,0) != 8 || cs(1,0) != 10 || cs(2,0) != 12 || cs(3,0) != 14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 8 10 12 14 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  8 || mat_(0,1) != 10 || mat_(0,2) != 12 || mat_(0,3) != 14 ||
             mat_(1,0) != 10 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) != -2 ||
             mat_(2,0) != 12 || mat_(2,1) !=  0 || mat_(2,2) !=  3 || mat_(2,3) !=  4 ||
             mat_(3,0) != 14 || mat_(3,1) != -2 || mat_(3,2) !=  4 || mat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  8 10 12 14 )\n"
                                        "( 10  1  0 -2 )\n"
                                        "( 12  0  3  4 )\n"
                                        "( 14 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         auto cs = blaze::columns( mat_, { 0UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( cs(0,0) != 6 || cs(1,0) != 7 || cs(2,0) != 8 || cs(3,0) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) != 6 || mat_(0,1) !=  7 || mat_(0,2) != 8 || mat_(0,3) !=  9 ||
             mat_(1,0) != 7 || mat_(1,1) !=  1 || mat_(1,2) != 0 || mat_(1,3) != -2 ||
             mat_(2,0) != 8 || mat_(2,1) !=  0 || mat_(2,2) != 3 || mat_(2,3) !=  4 ||
             mat_(3,0) != 9 || mat_(3,1) != -2 || mat_(3,2) != 4 || mat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n( 6  7  8  9 )\n"
                                        "( 7  1  0 -2 )\n"
                                        "( 8  0  3  4 )\n"
                                        "( 9 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         auto cs = blaze::columns( mat_, { 0UL } );
         int value = 1;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( cs(0,0) != 6 || cs(1,0) != 14 || cs(2,0) != 24 || cs(3,0) != 36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 14 24 36 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  6 || mat_(0,1) != 14 || mat_(0,2) != 24 || mat_(0,3) != 36 ||
             mat_(1,0) != 14 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) != -2 ||
             mat_(2,0) != 24 || mat_(2,1) !=  0 || mat_(2,2) !=  3 || mat_(2,3) !=  4 ||
             mat_(3,0) != 36 || mat_(3,1) != -2 || mat_(3,2) !=  4 || mat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  6 14 24 36 )\n"
                                        "( 14  1  0 -2 )\n"
                                        "( 24  0  3  4 )\n"
                                        "( 36 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         auto cs = blaze::columns( mat_, { 0UL } );

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( cs(0,0) != 3 || cs(1,0) != 7 || cs(2,0) != 12 || cs(3,0) != 18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 14 24 36 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( mat_(0,0) !=  3 || mat_(0,1) !=  7 || mat_(0,2) != 12 || mat_(0,3) != 18 ||
             mat_(1,0) !=  7 || mat_(1,1) !=  1 || mat_(1,2) !=  0 || mat_(1,3) != -2 ||
             mat_(2,0) != 12 || mat_(2,1) !=  0 || mat_(2,2) !=  3 || mat_(2,3) !=  4 ||
             mat_(3,0) != 18 || mat_(3,1) != -2 || mat_(3,2) !=  4 || mat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << mat_ << "\n"
                << "   Expected result:\n(  3  7 12 18 )\n"
                                        "(  7  1  0 -2 )\n"
                                        "( 12  0  3  4 )\n"
                                        "( 18 -2  4  5 )\n";
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
         test_ = "Row-major Iterator default constructor";

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
         test_ = "Row-major ConstIterator default constructor";

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
         test_ = "Row-major Iterator/ConstIterator conversion";

         auto cs = blaze::columns( tmat_, { 2UL } );
         auto it( begin( cs, 0UL ) );

         if( it == end( cs, 0UL ) || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Failed iterator conversion detected\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Counting the number of elements in 1st column via Iterator (end-begin)
      {
         test_ = "Row-major Iterator subtraction (end-begin)";

         auto cs = blaze::columns( tmat_, { 1UL } );
         const ptrdiff_t number( end( cs, 0UL ) - begin( cs, 0UL ) );

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

      // Counting the number of elements in 1st column via Iterator (begin-end)
      {
         test_ = "Row-major Iterator subtraction (begin-end)";

         auto cs = blaze::columns( tmat_, { 1UL } );
         const ptrdiff_t number( begin( cs, 0UL ) - end( cs, 0UL ) );

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

      // Counting the number of elements in 2nd column via ConstIterator (end-begin)
      {
         test_ = "Row-major ConstIterator subtraction (end-begin)";

         auto cs = blaze::columns( tmat_, { 2UL } );
         const ptrdiff_t number( cend( cs, 0UL ) - cbegin( cs, 0UL ) );

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

      // Counting the number of elements in 2nd column via ConstIterator (begin-end)
      {
         test_ = "Row-major ConstIterator subtraction (begin-end)";

         auto cs = blaze::columns( tmat_, { 2UL } );
         const ptrdiff_t number( cbegin( cs, 0UL ) - cend( cs, 0UL ) );

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

         auto cs = blaze::columns( tmat_, { 3UL } );
         auto it ( cbegin( cs, 0UL ) );
         auto end( cend( cs, 0UL ) );

         if( it == end || *it != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Invalid initial iterator detected\n";
            throw std::runtime_error( oss.str() );
         }

         ++it;

         if( it == end || *it != -2 ) {
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

         if( it == end || *it != -2 ) {
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

         if( it == end || *it != 4 ) {
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

         if( it == end || *it != 5 ) {
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

         auto cs = blaze::columns( tmat_, { 0UL } );
         int value = 6;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it = value++;
         }

         if( cs(0,0) != 6 || cs(1,0) != 7 || cs(2,0) != 8 || cs(3,0) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 6 || tmat_(0,1) !=  7 || tmat_(0,2) != 8 || tmat_(0,3) !=  9 ||
             tmat_(1,0) != 7 || tmat_(1,1) !=  1 || tmat_(1,2) != 0 || tmat_(1,3) != -2 ||
             tmat_(2,0) != 8 || tmat_(2,1) !=  0 || tmat_(2,2) != 3 || tmat_(2,3) !=  4 ||
             tmat_(3,0) != 9 || tmat_(3,1) != -2 || tmat_(3,2) != 4 || tmat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 6  7  8  9 )\n"
                                        "( 7  1  0 -2 )\n"
                                        "( 8  0  3  4 )\n"
                                        "( 9 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing addition assignment via Iterator
      {
         test_ = "Row-major addition assignment via Iterator";

         auto cs = blaze::columns( tmat_, { 0UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it += value++;
         }

         if( cs(0,0) != 8 || cs(1,0) != 10 || cs(2,0) != 12 || cs(3,0) != 14 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 8 10 12 14 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  8 || tmat_(0,1) != 10 || tmat_(0,2) != 12 || tmat_(0,3) != 14 ||
             tmat_(1,0) != 10 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) != -2 ||
             tmat_(2,0) != 12 || tmat_(2,1) !=  0 || tmat_(2,2) !=  3 || tmat_(2,3) !=  4 ||
             tmat_(3,0) != 14 || tmat_(3,1) != -2 || tmat_(3,2) !=  4 || tmat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  8 10 12 14 )\n"
                                        "( 10  1  0 -2 )\n"
                                        "( 12  0  3  4 )\n"
                                        "( 14 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing subtraction assignment via Iterator
      {
         test_ = "Row-major subtraction assignment via Iterator";

         auto cs = blaze::columns( tmat_, { 0UL } );
         int value = 2;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it -= value++;
         }

         if( cs(0,0) != 6 || cs(1,0) != 7 || cs(2,0) != 8 || cs(3,0) != 9 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 7 8 9 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) != 6 || tmat_(0,1) !=  7 || tmat_(0,2) != 8 || tmat_(0,3) !=  9 ||
             tmat_(1,0) != 7 || tmat_(1,1) !=  1 || tmat_(1,2) != 0 || tmat_(1,3) != -2 ||
             tmat_(2,0) != 8 || tmat_(2,1) !=  0 || tmat_(2,2) != 3 || tmat_(2,3) !=  4 ||
             tmat_(3,0) != 9 || tmat_(3,1) != -2 || tmat_(3,2) != 4 || tmat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n( 6  7  8  9 )\n"
                                        "( 7  1  0 -2 )\n"
                                        "( 8  0  3  4 )\n"
                                        "( 9 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing multiplication assignment via Iterator
      {
         test_ = "Row-major multiplication assignment via Iterator";

         auto cs = blaze::columns( tmat_, { 0UL } );
         int value = 1;

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it *= value++;
         }

         if( cs(0,0) != 6 || cs(1,0) != 14 || cs(2,0) != 24 || cs(3,0) != 36 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 14 24 36 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  6 || tmat_(0,1) != 14 || tmat_(0,2) != 24 || tmat_(0,3) != 36 ||
             tmat_(1,0) != 14 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) != -2 ||
             tmat_(2,0) != 24 || tmat_(2,1) !=  0 || tmat_(2,2) !=  3 || tmat_(2,3) !=  4 ||
             tmat_(3,0) != 36 || tmat_(3,1) != -2 || tmat_(3,2) !=  4 || tmat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  6 14 24 36 )\n"
                                        "( 14  1  0 -2 )\n"
                                        "( 24  0  3  4 )\n"
                                        "( 36 -2  4  5 )\n";
            throw std::runtime_error( oss.str() );
         }
      }

      // Testing division assignment via Iterator
      {
         test_ = "Row-major division assignment via Iterator";

         auto cs = blaze::columns( tmat_, { 0UL } );

         for( auto it=begin( cs, 0UL ); it!=end( cs, 0UL ); ++it ) {
            *it /= 2;
         }

         if( cs(0,0) != 3 || cs(1,0) != 7 || cs(2,0) != 12 || cs(3,0) != 18 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << cs << "\n"
                << "   Expected result:\n( 6 14 24 36 )\n";
            throw std::runtime_error( oss.str() );
         }

         if( tmat_(0,0) !=  3 || tmat_(0,1) !=  7 || tmat_(0,2) != 12 || tmat_(0,3) != 18 ||
             tmat_(1,0) !=  7 || tmat_(1,1) !=  1 || tmat_(1,2) !=  0 || tmat_(1,3) != -2 ||
             tmat_(2,0) != 12 || tmat_(2,1) !=  0 || tmat_(2,2) !=  3 || tmat_(2,3) !=  4 ||
             tmat_(3,0) != 18 || tmat_(3,1) != -2 || tmat_(3,2) !=  4 || tmat_(3,3) !=  5 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Assignment via iterator failed\n"
                << " Details:\n"
                << "   Result:\n" << tmat_ << "\n"
                << "   Expected result:\n(  3  7 12 18 )\n"
                                        "(  7  1  0 -2 )\n"
                                        "( 12  0  3  4 )\n"
                                        "( 18 -2  4  5 )\n";
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
void DenseSymmetricTest::testNonZeros()
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
void DenseSymmetricTest::testReset()
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
void DenseSymmetricTest::testClear()
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
/*!\brief Test of the \c isDefault() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c isDefault() function with the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseSymmetricTest::testIsDefault()
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
void DenseSymmetricTest::testIsSame()
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
/*!\brief Test of the \c submatrix() function with the Columns class template.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the \c submatrix() function with the Columns specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseSymmetricTest::testSubmatrix()
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
void DenseSymmetricTest::testRow()
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
void DenseSymmetricTest::testRows()
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

         if( *rs.begin( 2UL ) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs.begin( 2UL ) << "\n"
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

         if( *rs.begin( 2UL ) != 3 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *rs.begin( 2UL ) << "\n"
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
void DenseSymmetricTest::testColumn()
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

         if( *col1.begin() != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *col1.begin() << "\n"
                << "   Expected result: 0\n";
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

         if( *col1.begin() != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *col1.begin() << "\n"
                << "   Expected result: 0\n";
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
void DenseSymmetricTest::testColumns()
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

         if( *cs2.begin( 2UL ) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *cs2.begin( 2UL ) << "\n"
                << "   Expected result: 0\n";
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

         if( *cs2.begin( 2UL ) != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *cs2.begin( 2UL ) << "\n"
                << "   Expected result: 0\n";
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
void DenseSymmetricTest::testBand()
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

         if( *b1.begin() != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *b1.begin() << "\n"
                << "   Expected result: 0\n";
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

         if( *b1.begin() != 0 ) {
            std::ostringstream oss;
            oss << " Test: " << test_ << "\n"
                << " Error: Iterator access failed\n"
                << " Details:\n"
                << "   Result: " << *b1.begin() << "\n"
                << "   Expected result: 0\n";
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
void DenseSymmetricTest::initialize()
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
   std::cout << "   Running Columns dense symmetric test..." << std::endl;

   try
   {
      RUN_COLUMNS_DENSESYMMETRIC_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during Columns dense symmetric test:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
