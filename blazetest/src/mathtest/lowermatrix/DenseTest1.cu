//=================================================================================================
/*!
//  \file src/mathtest/lowermatrix/DenseTest1.cpp
//  \brief Source file for the LowerMatrix dense test (part 1)
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
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/CustomMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/HybridMatrix.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/util/policies/ArrayDelete.h>
#include <blazetest/mathtest/lowermatrix/DenseTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


namespace blazetest {

namespace mathtest {

namespace lowermatrix {

//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Constructor for the LowerMatrix dense test.
//
// \exception std::runtime_error Operation error detected.
*/
DenseTest::DenseTest()
{
   testConstructors();
   testAssignment();
   testAddAssign();
   testSubAssign();
   testSchurAssign();
}
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the LowerMatrix constructors.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all constructors of the LowerMatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testConstructors()
{
   //=====================================================================================
   // Row-major default constructor
   //=====================================================================================

   // Default constructor (StaticMatrix)
   {
      test_ = "Row-major LowerMatrix default constructor (StaticMatrix)";

      const blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > lower;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 0UL );
   }

   // Default constructor (HybridMatrix)
   {
      test_ = "Row-major LowerMatrix default constructor (HybridMatrix)";

      const blaze::LowerMatrix< blaze::HybridMatrix<int,3UL,3UL,blaze::rowMajor> > lower;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Default constructor (DynamicMatrix)
   {
      test_ = "Row-major LowerMatrix default constructor (DynamicMatrix)";

      const LT lower;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }


   //=====================================================================================
   // Row-major single argument constructor
   //=====================================================================================

   // Single argument constructor (StaticMatrix)
   {
      test_ = "Row-major LowerMatrix single argument constructor (StaticMatrix)";

      const blaze::LowerMatrix< blaze::StaticMatrix<int,2UL,2UL,blaze::rowMajor> > lower( 5 );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 5 || lower(0,1) != 0 ||
          lower(1,0) != 5 || lower(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 5 0 )\n( 5 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (HybridMatrix)
   {
      test_ = "Row-major LowerMatrix single argument constructor (HybridMatrix)";

      const blaze::LowerMatrix< blaze::HybridMatrix<int,3UL,3UL,blaze::rowMajor> > lower( 2UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 0UL );

      if( lower(0,0) != 0 || lower(0,1) != 0 ||
          lower(1,0) != 0 || lower(1,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (DynamicMatrix)
   {
      test_ = "Row-major LowerMatrix single argument constructor (DynamicMatrix)";

      const LT lower( 2UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 0UL );

      if( lower(0,0) != 0 || lower(0,1) != 0 ||
          lower(1,0) != 0 || lower(1,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (0x0)
   {
      test_ = "Row-major LowerMatrix single argument constructor (0x0)";

      const blaze::DynamicMatrix<int,blaze::rowMajor> mat;
      const LT lower( mat );

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Single argument constructor (lower)
   {
      test_ = "Row-major LowerMatrix single argument constructor (lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> mat;
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      const LT lower( mat );

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 2UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (non-lower)
   {
      test_ = "Row-major LowerMatrix single argument constructor (non-lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> mat;
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         const LT lower( mat );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of non-lower LowerMatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Single argument constructor (LowerMatrix)
   {
      test_ = "Row-major LowerMatrix single argument constructor (LowerMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > lower1;
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      const LT lower2( lower1 );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major two argument constructor
   //=====================================================================================

   // Two argument constructor (HybridMatrix)
   {
      test_ = "Row-major LowerMatrix two argument constructor (HybridMatrix)";

      const blaze::LowerMatrix< blaze::HybridMatrix<int,3UL,3UL,blaze::rowMajor> > lower( 2UL, 5 );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 5 || lower(0,1) != 0 ||
          lower(1,0) != 5 || lower(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 5 0 )\n( 5 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Two argument constructor (DynamicMatrix)
   {
      test_ = "Row-major LowerMatrix two argument constructor (DynamicMatrix)";

      const LT lower( 2UL, 5 );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 5 || lower(0,1) != 0 ||
          lower(1,0) != 5 || lower(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 5 0 )\n( 5 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major list initialization
   //=====================================================================================

   // Complete initializer list
   {
      test_ = "Row-major LowerMatrix initializer list constructor (complete list)";

      const LT lower{ { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Incomplete initializer list
   {
      test_ = "Row-major LowerMatrix initializer list constructor (incomplete list)";

      const LT lower{ { 1 }, { 2, 3 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major array initialization
   //=====================================================================================

   // Dynamic array initialization constructor
   {
      test_ = "Row-major LowerMatrix dynamic array initialization constructor";

      std::unique_ptr<int[]> array( new int[9] );
      array[0] = 1;
      array[1] = 0;
      array[2] = 0;
      array[3] = 2;
      array[4] = 3;
      array[5] = 0;
      array[6] = 4;
      array[7] = 5;
      array[8] = 6;
      const LT lower( 3UL, array.get() );

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Static array initialization constructor
   {
      test_ = "Row-major LowerMatrix static array initialization constructor";

      const int array[3][3] = { { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };
      const LT lower( array );

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major custom matrix constructors
   //=====================================================================================

   // Custom matrix constructor (ElementType*, size_t)
   {
      test_ = "Row-major LowerMatrix custom matrix constructor (ElementType*, size_t)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[5UL] );
      memory[1] = 1;
      memory[2] = 0;
      memory[3] = 2;
      memory[4] = 3;
      const blaze::LowerMatrix<UnalignedUnpadded> lower( memory.get()+1UL, 2UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 )\n( 2 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Custom matrix constructor (ElementType*, size_t, size_t)
   {
      test_ = "Row-major LowerMatrix custom matrix constructor (ElementType*, size_t, size_t)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::rowMajor;

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,rowMajor>;
      std::unique_ptr<int[]> memory( new int[11UL] );
      memory[1] = 1;
      memory[2] = 0;
      memory[6] = 2;
      memory[7] = 3;
      const blaze::LowerMatrix<UnalignedUnpadded> lower( memory.get()+1UL, 2UL, 5UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 )\n( 2 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy constructor
   //=====================================================================================

   // Copy constructor (0x0)
   {
      test_ = "Row-major LowerMatrix copy constructor (0x0)";

      const LT lower1;
      const LT lower2( lower1 );

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Copy constructor (3x3)
   {
      test_ = "Row-major LowerMatrix copy constructor (3x3)";

      LT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      const LT lower2( lower1 );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major move constructor
   //=====================================================================================

   // Move constructor (0x0)
   {
      test_ = "Row-major LowerMatrix move constructor (0x0)";

      LT lower1;
      LT lower2( std::move( lower1 ) );

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Move constructor (3x3)
   {
      test_ = "Row-major LowerMatrix move constructor (3x3)";

      LT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      LT lower2( std::move( lower1 ) );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major default constructor
   //=====================================================================================

   // Default constructor (StaticMatrix)
   {
      test_ = "Column-major LowerMatrix default constructor (StaticMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> > lower;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 0UL );
   }

   // Default constructor (HybridMatrix)
   {
      test_ = "Column-major LowerMatrix default constructor (HybridMatrix)";

      blaze::LowerMatrix< blaze::HybridMatrix<int,3UL,3UL,blaze::columnMajor> > lower;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Default constructor (DynamicMatrix)
   {
      test_ = "Column-major LowerMatrix default constructor (DynamicMatrix)";

      OLT lower;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }


   //=====================================================================================
   // Column-major single argument constructor
   //=====================================================================================

   // Single argument constructor (StaticMatrix)
   {
      test_ = "Column-major LowerMatrix single argument constructor (StaticMatrix)";

      const blaze::LowerMatrix< blaze::StaticMatrix<int,2UL,2UL,blaze::columnMajor> > lower( 5 );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 5 || lower(0,1) != 0 ||
          lower(1,0) != 5 || lower(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 5 0 )\n( 5 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (HybridMatrix)
   {
      test_ = "Column-major LowerMatrix single argument constructor (HybridMatrix)";

      const blaze::LowerMatrix< blaze::HybridMatrix<int,3UL,3UL,blaze::columnMajor> > lower( 2UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 0UL );

      if( lower(0,0) != 0 || lower(0,1) != 0 ||
          lower(1,0) != 0 || lower(1,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (DynamicMatrix)
   {
      test_ = "Column-major LowerMatrix single argument constructor (DynamicMatrix)";

      const OLT lower( 2UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 0UL );

      if( lower(0,0) != 0 || lower(0,1) != 0 ||
          lower(1,0) != 0 || lower(1,1) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 0 0 )\n( 0 0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (0x0)
   {
      test_ = "Column-major LowerMatrix single argument constructor (0x0)";

      const blaze::DynamicMatrix<int,blaze::columnMajor> mat;
      const OLT lower( mat );

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Single argument constructor (lower)
   {
      test_ = "Column-major LowerMatrix single argument constructor (lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> mat;
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      const OLT lower( mat );

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Single argument constructor (non-lower)
   {
      test_ = "Column-major LowerMatrix single argument constructor (non-lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> mat;
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         const OLT lower( mat );

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Setup of non-lower LowerMatrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Single argument constructor (LowerMatrix)
   {
      test_ = "Column-major LowerMatrix single argument constructor (LowerMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> > lower1;
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      const OLT lower2( lower1 );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major two argument constructor
   //=====================================================================================

   // Two argument constructor (HybridMatrix)
   {
      test_ = "Column-major LowerMatrix two argument constructor (HybridMatrix)";

      const blaze::LowerMatrix< blaze::HybridMatrix<int,3UL,3UL,blaze::columnMajor> > lower( 2UL, 5 );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 5 || lower(0,1) != 0 ||
          lower(1,0) != 5 || lower(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 5 0 )\n( 5 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Two argument constructor (DynamicMatrix)
   {
      test_ = "Column-major LowerMatrix two argument constructor (DynamicMatrix)";

      const OLT lower( 2UL, 5 );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 5 || lower(0,1) != 0 ||
          lower(1,0) != 5 || lower(1,1) != 5 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 5 0 )\n( 5 5 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major list initialization
   //=====================================================================================

   // Complete initializer list
   {
      test_ = "Column-major LowerMatrix initializer list constructor (complete list)";

      const OLT lower{ { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Incomplete initializer list
   {
      test_ = "Column-major LowerMatrix initializer list constructor (incomplete list)";

      const OLT lower{ { 1 }, { 2, 3 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major array initialization
   //=====================================================================================

   // Dynamic array initialization constructor
   {
      test_ = "Column-major LowerMatrix dynamic array initialization constructor";

      std::unique_ptr<int[]> array( new int[9] );
      array[0] = 1;
      array[1] = 2;
      array[2] = 4;
      array[3] = 0;
      array[4] = 3;
      array[5] = 5;
      array[6] = 0;
      array[7] = 0;
      array[8] = 6;
      const OLT lower( 3UL, array.get() );

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Static array initialization constructor
   {
      test_ = "Column-major LowerMatrix static array initialization constructor";

      const int array[3][3] = { { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };
      const OLT lower( array );

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major custom matrix constructors
   //=====================================================================================

   // Custom matrix constructor (ElementType*, size_t)
   {
      test_ = "Column-major LowerMatrix custom matrix constructor (ElementType*, size_t)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::columnMajor;

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
      std::unique_ptr<int[]> memory( new int[5UL] );
      memory[1] = 1;
      memory[2] = 2;
      memory[3] = 0;
      memory[4] = 3;
      const blaze::LowerMatrix<UnalignedUnpadded> lower( memory.get()+1UL, 2UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 )\n( 2 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Custom matrix constructor (ElementType*, size_t, size_t)
   {
      test_ = "Column-major LowerMatrix custom matrix constructor (ElementType*, size_t, size_t)";

      using blaze::unaligned;
      using blaze::unpadded;
      using blaze::columnMajor;

      using UnalignedUnpadded = blaze::CustomMatrix<int,unaligned,unpadded,columnMajor>;
      std::unique_ptr<int[]> memory( new int[11UL] );
      memory[1] = 1;
      memory[2] = 2;
      memory[6] = 0;
      memory[7] = 3;
      const blaze::LowerMatrix<UnalignedUnpadded> lower( memory.get()+1UL, 2UL, 5UL );

      checkRows    ( lower, 2UL );
      checkColumns ( lower, 2UL );
      checkCapacity( lower, 4UL );
      checkNonZeros( lower, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 )\n( 2 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major copy constructor
   //=====================================================================================

   // Copy constructor (0x0)
   {
      test_ = "Column-major LowerMatrix copy constructor (0x0)";

      const OLT lower1;
      const OLT lower2( lower1 );

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Copy constructor (3x3)
   {
      test_ = "Column-major LowerMatrix copy constructor (3x3)";

      OLT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      const OLT lower2( lower1 );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major move constructor
   //=====================================================================================

   // Move constructor (0x0)
   {
      test_ = "Column-major LowerMatrix move constructor (0x0)";

      OLT lower1;
      OLT lower2( std::move( lower1 ) );

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Move constructor (3x3)
   {
      test_ = "Column-major LowerMatrix move constructor (3x3)";

      OLT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      OLT lower2( std::move( lower1 ) );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Construction failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the LowerMatrix assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of all assignment operators of the LowerMatrix specialization.
// In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAssignment()
{
   //=====================================================================================
   // Row-major homogeneous assignment
   //=====================================================================================

   // Homogeneous assignment (3x3)
   {
      test_ = "Row-major LowerMatrix homogeneous assignment (3x3)";

      LT lower( 3UL );
      lower = 2;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) != 2 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) != 2 || lower(2,1) != 2 || lower(2,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 2 0 0 )\n( 2 2 0 )\n( 2 2 2 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major list assignment
   //=====================================================================================

   // Complete initializer list
   {
      test_ = "Row-major LowerMatrix initializer list assignment (complete list)";

      LT lower;
      lower = { { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Incomplete initializer list
   {
      test_ = "Row-major LowerMatrix initializer list assignment (incomplete list)";

      LT lower;
      lower = { { 1 }, { 2, 3 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major array assignment
   //=====================================================================================

   // Array assignment
   {
      test_ = "Row-major LowerMatrix array assignment";

      const int array[3][3] = { { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };
      LT lower;
      lower = array;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major copy assignment
   //=====================================================================================

   // Copy assignment (0x0)
   {
      test_ = "Row-major LowerMatrix copy assignment (0x0)";

      LT lower1, lower2;

      lower2 = lower1;

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Copy assignment (3x3)
   {
      test_ = "Row-major LowerMatrix copy assignment (3x3)";

      LT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,1) =  0;
      lower1(2,2) =  3;

      LT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major move assignment
   //=====================================================================================

   // Move assignment (0x0)
   {
      test_ = "Row-major LowerMatrix move assignment (0x0)";

      LT lower1, lower2;

      lower2 = std::move( lower1 );

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Move assignment (3x3)
   {
      test_ = "Row-major LowerMatrix move assignment (3x3)";

      LT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,1) =  0;
      lower1(2,2) =  3;

      LT lower2;
      lower2 = std::move( lower1 );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major dense matrix assignment
   //=====================================================================================

   // Conversion assignment (0x0)
   {
      test_ = "Row-major LowerMatrix dense matrix assignment (0x0)";

      const blaze::DynamicMatrix<int,blaze::rowMajor> mat;

      LT lower;
      lower = mat;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Row-major/row-major dense matrix assignment (lower)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix assignment (lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> mat;
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      LT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 2UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix assignment (lower)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix assignment (lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> mat;
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      LT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 2UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major dense matrix assignment (non-lower)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix assignment (non-lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> mat;
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         LT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/column-major dense matrix assignment (non-lower)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix assignment (non-lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> mat;
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         LT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/row-major dense matrix assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > lower1;
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      LT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> > lower1;
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      LT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major sparse matrix assignment
   //=====================================================================================

   // Conversion assignment (0x0)
   {
      test_ = "Row-major LowerMatrix sparse matrix assignment (0x0)";

      const blaze::CompressedMatrix<int,blaze::rowMajor> mat;

      LT lower;
      lower = mat;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Row-major/row-major sparse matrix assignment (lower)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix assignment (lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;
      mat.insert( 1UL, 2UL, 0 );

      LT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 2UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix assignment (lower)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix assignment (lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;
      mat.insert( 1UL, 2UL, 0 );

      LT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 2UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major sparse matrix assignment (non-lower)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         LT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/column-major sparse matrix assignment (non-lower)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         LT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/row-major sparse matrix assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 5UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      LT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 5UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      LT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 2UL );
      checkNonZeros( lower2, 2UL, 2UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major homogeneous assignment
   //=====================================================================================

   // Homogeneous assignment (3x3)
   {
      test_ = "Column-major LowerMatrix homogeneous assignment (3x3)";

      OLT lower( 3UL );
      lower = 2;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) != 2 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) != 2 || lower(2,1) != 2 || lower(2,2) != 2 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 2 0 0 )\n( 2 2 0 )\n( 2 2 2 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major list assignment
   //=====================================================================================

   // Complete initializer list
   {
      test_ = "Column-major LowerMatrix initializer list assignment (complete list)";

      OLT lower;
      lower = { { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Incomplete initializer list
   {
      test_ = "Column-major LowerMatrix initializer list assignment (incomplete list)";

      OLT lower;
      lower = { { 1 }, { 2, 3 }, { 4, 5, 6 } };

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major array assignment
   //=====================================================================================

   // Array assignment
   {
      test_ = "Column-major LowerMatrix array assignment";

      const int array[3][3] = { { 1, 0, 0 }, { 2, 3, 0 }, { 4, 5, 6 } };
      OLT lower;
      lower = array;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 6UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 2UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) != 1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != 2 || lower(1,1) != 3 || lower(1,2) != 0 ||
          lower(2,0) != 4 || lower(2,1) != 5 || lower(2,2) != 6 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n( 1 0 0 )\n( 2 3 0 )\n( 4 5 6 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major copy assignment
   //=====================================================================================

   // Copy assignment (0x0)
   {
      test_ = "Column-major LowerMatrix copy assignment (0x0)";

      OLT lower1, lower2;

      lower2 = lower1;

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Copy assignment (3x3)
   {
      test_ = "Column-major LowerMatrix copy assignment (3x3)";

      OLT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,1) =  0;
      lower1(2,2) =  3;

      OLT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major move assignment
   //=====================================================================================

   // Move assignment (0x0)
   {
      test_ = "Column-major LowerMatrix move assignment (0x0)";

      OLT lower1, lower2;

      lower2 = std::move( lower1 );

      checkRows    ( lower2, 0UL );
      checkColumns ( lower2, 0UL );
      checkNonZeros( lower2, 0UL );
   }

   // Move assignment (3x3)
   {
      test_ = "Column-major LowerMatrix move assignment (3x3)";

      OLT lower1( 3UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,1) =  0;
      lower1(2,2) =  3;

      OLT lower2;
      lower2 = std::move( lower1 );

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major dense matrix assignment
   //=====================================================================================

   // Conversion assignment (0x0)
   {
      test_ = "Column-major LowerMatrix dense matrix assignment (0x0)";

      const blaze::DynamicMatrix<int,blaze::columnMajor> mat;

      OLT lower;
      lower = mat;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Column-major/row-major dense matrix assignment (lower)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix assignment (lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> mat;
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      OLT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix assignment (lower)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix assignment (lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> mat;
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      OLT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major dense matrix assignment (non-lower)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix assignment (non-lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> mat;
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         OLT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/column-major dense matrix assignment (non-lower)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix assignment (non-lower)";

      blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> mat;
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         OLT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/row-major dense matrix assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::rowMajor> > lower1;
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      OLT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::StaticMatrix<int,3UL,3UL,blaze::columnMajor> > lower1;
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      OLT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major sparse matrix assignment
   //=====================================================================================

   // Conversion assignment (0x0)
   {
      test_ = "Column-major LowerMatrix sparse matrix assignment (0x0)";

      const blaze::CompressedMatrix<int,blaze::columnMajor> mat;

      OLT lower;
      lower = mat;

      checkRows    ( lower, 0UL );
      checkColumns ( lower, 0UL );
      checkNonZeros( lower, 0UL );
   }

   // Column-major/row-major sparse matrix assignment (lower)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix assignment (lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix assignment (lower)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix assignment (lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower;
      lower = mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -4 || lower(1,1) != 2 || lower(1,2) != 0 ||
          lower(2,0) !=  7 || lower(2,1) != 0 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major sparse matrix assignment (non-lower)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         OLT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/column-major sparse matrix assignment (non-lower)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  1;
      mat(0,2) =  5;
      mat(1,0) = -4;
      mat(1,1) =  2;
      mat(2,0) =  7;
      mat(2,2) =  3;

      try {
         OLT lower;
         lower = mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/row-major sparse matrix assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 5UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      OLT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 5UL );
      lower1(0,0) =  1;
      lower1(1,0) = -4;
      lower1(1,1) =  2;
      lower1(2,0) =  7;
      lower1(2,2) =  3;

      OLT lower2;
      lower2 = lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -4 || lower2(1,1) != 2 || lower2(1,2) != 0 ||
          lower2(2,0) !=  7 || lower2(2,1) != 0 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -4 2 0 )\n(  7 0 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the LowerMatrix addition assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the addition assignment operators of the LowerMatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testAddAssign()
{
   //=====================================================================================
   // Row-major dense matrix addition assignment
   //=====================================================================================

   // Row-major/row-major dense matrix addition assignment (lower)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix addition assignment (lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix addition assignment (lower)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix addition assignment (lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major dense matrix addition assignment (non-lower)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix addition assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/column-major dense matrix addition assignment (non-lower)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix addition assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/row-major dense matrix addition assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix addition assignment (LowerMatrix)";

      LT lower1( 3UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix addition assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix addition assignment (LowerMatrix)";

      OLT lower1( 3UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major sparse matrix addition assignment
   //=====================================================================================

   // Row-major/row-major sparse matrix addition assignment (lower)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix addition assignment (lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix addition assignment (lower)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix addition assignment (lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major sparse matrix addition assignment (non-lower)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix addition assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/column-major sparse matrix addition assignment (non-lower)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix addition assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/row-major sparse matrix addition assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix addition assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 4UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix addition assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix addition assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 4UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major dense matrix addition assignment
   //=====================================================================================

   // Column-major/row-major dense matrix addition assignment (lower)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix addition assignment (lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix addition assignment (lower)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix addition assignment (lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major dense matrix addition assignment (non-lower)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix addition assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/column-major dense matrix addition assignment (non-lower)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix addition assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/row-major dense matrix addition assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix addition assignment (LowerMatrix)";

      LT lower1( 3UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix addition assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix addition assignment (LowerMatrix)";

      OLT lower1( 3UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major sparse matrix addition assignment
   //=====================================================================================

   // Column-major/row-major sparse matrix addition assignment (lower)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix addition assignment (lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix addition assignment (lower)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix addition assignment (lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) =  2;
      mat(1,1) = -2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower += mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) != 0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) != 0 || lower(1,2) != 0 ||
          lower(2,0) != 13 || lower(2,1) != 5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major sparse matrix addition assignment (non-lower)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix addition assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/column-major sparse matrix addition assignment (non-lower)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix addition assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower += mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/row-major sparse matrix addition assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix addition assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 4UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix addition assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix addition assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 4UL );
      lower1(1,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 += lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) != 0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) != 0 || lower2(1,2) != 0 ||
          lower2(2,0) != 13 || lower2(2,1) != 5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Addition assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1 0 0 )\n( -2 0 0 )\n( 13 5 3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the LowerMatrix subtraction assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the subtraction assignment operators of the LowerMatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testSubAssign()
{
   //=====================================================================================
   // Row-major dense matrix subtraction assignment
   //=====================================================================================

   // Row-major/row-major dense matrix subtraction assignment (lower)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix subtraction assignment (lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix subtraction assignment (lower)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix subtraction assignment (lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major dense matrix subtraction assignment (non-lower)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix subtraction assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/column-major dense matrix subtraction assignment (non-lower)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix subtraction assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/row-major dense matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix subtraction assignment (LowerMatrix)";

      LT lower1( 3UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix subtraction assignment (LowerMatrix)";

      OLT lower1( 3UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major sparse matrix subtraction assignment
   //=====================================================================================

   // Row-major/row-major sparse matrix subtraction assignment (lower)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix subtraction assignment (lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix subtraction assignment (lower)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix subtraction assignment (lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 3UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major sparse matrix subtraction assignment (non-lower)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix subtraction assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/column-major sparse matrix subtraction assignment (non-lower)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix subtraction assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Row-major/row-major sparse matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix subtraction assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 4UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix subtraction assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 4UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 3UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major dense matrix subtraction assignment
   //=====================================================================================

   // Column-major/row-major dense matrix subtraction assignment (lower)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix subtraction assignment (lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix subtraction assignment (lower)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix subtraction assignment (lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major dense matrix subtraction assignment (non-lower)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix subtraction assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/column-major dense matrix subtraction assignment (non-lower)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix subtraction assignment (non-lower)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 0 );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/row-major dense matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix subtraction assignment (LowerMatrix)";

      LT lower1( 3UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix subtraction assignment (LowerMatrix)";

      OLT lower1( 3UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major sparse matrix subtraction assignment
   //=====================================================================================

   // Column-major/row-major sparse matrix subtraction assignment (lower)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix subtraction assignment (lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix subtraction assignment (lower)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix subtraction assignment (lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 5UL );
      mat(1,0) = -2;
      mat(1,1) =  2;
      mat(2,0) =  6;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower -= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 5UL );
      checkNonZeros( lower, 0UL, 3UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  1 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) != -2 || lower(1,1) !=  0 || lower(1,2) != 0 ||
          lower(2,0) !=  1 || lower(2,1) != -5 || lower(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major sparse matrix subtraction assignment (non-lower)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix subtraction assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower row-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/column-major sparse matrix subtraction assignment (non-lower)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix subtraction assignment (non-lower)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 1UL );
      mat(0,2) = 6;

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      try {
         lower -= mat;

         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment of non-lower column-major matrix succeeded\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n";
         throw std::runtime_error( oss.str() );
      }
      catch( std::invalid_argument& ) {}
   }

   // Column-major/row-major sparse matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix subtraction assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 4UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix subtraction assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix subtraction assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 4UL );
      lower1(1,0) = -2;
      lower1(1,1) =  2;
      lower1(2,0) =  6;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 -= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 5UL );
      checkNonZeros( lower2, 0UL, 3UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  1 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) != -2 || lower2(1,1) !=  0 || lower2(1,2) != 0 ||
          lower2(2,0) !=  1 || lower2(2,1) != -5 || lower2(2,2) != 3 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Subtraction assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  1  0  0 )\n( -2  0  0 )\n(  1 -5  3 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Test of the LowerMatrix Schur product assignment operators.
//
// \return void
// \exception std::runtime_error Error detected.
//
// This function performs a test of the Schur product assignment operators of the LowerMatrix
// specialization. In case an error is detected, a \a std::runtime_error exception is thrown.
*/
void DenseTest::testSchurAssign()
{
   //=====================================================================================
   // Row-major dense matrix Schur product assignment
   //=====================================================================================

   // Row-major/row-major dense matrix Schur product assignment (general)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix Schur product assignment (general)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat{ { 2, 0, 9 }, { 0, -2, 0 }, { 3, 5, 0 } };

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix Schur product assignment (general)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix Schur product assignment (general)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat{ { 2, 0, 9 }, { 0, -2, 0 }, { 3, 5, 0 } };

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major dense matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix dense matrix Schur product assignment (LowerMatrix)";

      LT lower1( 3UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major dense matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix dense matrix Schur product assignment (LowerMatrix)";

      OLT lower1( 3UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Row-major sparse matrix Schur product assignment
   //=====================================================================================

   // Row-major/row-major sparse matrix Schur product assignment (general)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix Schur product assignment (general)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  2;
      mat(0,2) =  9;
      mat(1,1) = -2;
      mat(2,0) =  3;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix Schur product assignment (general)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix Schur product assignment (general)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  2;
      mat(0,2) =  9;
      mat(1,1) = -2;
      mat(2,0) =  3;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      LT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 1UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 1UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/row-major sparse matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Row-major/row-major LowerMatrix sparse matrix Schur product assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 4UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Row-major/column-major sparse matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Row-major/column-major LowerMatrix sparse matrix Schur product assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 4UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      LT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 1UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 1UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major dense matrix Schur product assignment
   //=====================================================================================

   // Column-major/row-major dense matrix Schur product assignment (general)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix Schur product assignment (general)";

      blaze::DynamicMatrix<int,blaze::rowMajor> mat{ { 2, 0, 9 }, { 0, -2, 0 }, { 3, 5, 0 } };

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 2UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 0UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix Schur product assignment (general)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix Schur product assignment (general)";

      blaze::DynamicMatrix<int,blaze::columnMajor> mat{ { 2, 0, 9 }, { 0, -2, 0 }, { 3, 5, 0 } };

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 2UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 0UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major dense matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix dense matrix Schur product assignment (LowerMatrix)";

      LT lower1( 3UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 2UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 0UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major dense matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix dense matrix Schur product assignment (LowerMatrix)";

      OLT lower1( 3UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 2UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 0UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }


   //=====================================================================================
   // Column-major sparse matrix Schur product assignment
   //=====================================================================================

   // Column-major/row-major sparse matrix Schur product assignment (general)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix Schur product assignment (general)";

      blaze::CompressedMatrix<int,blaze::rowMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  2;
      mat(0,2) =  9;
      mat(1,1) = -2;
      mat(2,0) =  3;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 2UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 0UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix Schur product assignment (general)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix Schur product assignment (general)";

      blaze::CompressedMatrix<int,blaze::columnMajor> mat( 3UL, 3UL, 6UL );
      mat(0,0) =  2;
      mat(0,2) =  9;
      mat(1,1) = -2;
      mat(2,0) =  3;
      mat(2,1) =  5;
      mat.insert( 1UL, 2UL, 0 );

      OLT lower( 3UL );
      lower(0,0) =  1;
      lower(1,0) = -4;
      lower(1,1) =  2;
      lower(2,0) =  7;
      lower(2,2) =  3;

      lower %= mat;

      checkRows    ( lower, 3UL );
      checkColumns ( lower, 3UL );
      checkCapacity( lower, 9UL );
      checkNonZeros( lower, 3UL );
      checkNonZeros( lower, 0UL, 2UL );
      checkNonZeros( lower, 1UL, 1UL );
      checkNonZeros( lower, 2UL, 0UL );

      if( lower(0,0) !=  2 || lower(0,1) !=  0 || lower(0,2) != 0 ||
          lower(1,0) !=  0 || lower(1,1) != -4 || lower(1,2) != 0 ||
          lower(2,0) != 21 || lower(2,1) !=  0 || lower(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/row-major sparse matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Column-major/row-major LowerMatrix sparse matrix Schur product assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::rowMajor> > lower1( 3UL, 4UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 2UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 0UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }

   // Column-major/column-major sparse matrix Schur product assignment (LowerMatrix)
   {
      test_ = "Column-major/column-major LowerMatrix sparse matrix Schur product assignment (LowerMatrix)";

      blaze::LowerMatrix< blaze::CompressedMatrix<int,blaze::columnMajor> > lower1( 3UL, 4UL );
      lower1(0,0) =  2;
      lower1(1,1) = -2;
      lower1(2,0) =  3;
      lower1(2,1) =  5;

      OLT lower2( 3UL );
      lower2(0,0) =  1;
      lower2(1,0) = -4;
      lower2(1,1) =  2;
      lower2(2,0) =  7;
      lower2(2,2) =  3;

      lower2 %= lower1;

      checkRows    ( lower2, 3UL );
      checkColumns ( lower2, 3UL );
      checkCapacity( lower2, 9UL );
      checkNonZeros( lower2, 3UL );
      checkNonZeros( lower2, 0UL, 2UL );
      checkNonZeros( lower2, 1UL, 1UL );
      checkNonZeros( lower2, 2UL, 0UL );

      if( lower2(0,0) !=  2 || lower2(0,1) !=  0 || lower2(0,2) != 0 ||
          lower2(1,0) !=  0 || lower2(1,1) != -4 || lower2(1,2) != 0 ||
          lower2(2,0) != 21 || lower2(2,1) !=  0 || lower2(2,2) != 0 ) {
         std::ostringstream oss;
         oss << " Test: " << test_ << "\n"
             << " Error: Schur product assignment failed\n"
             << " Details:\n"
             << "   Result:\n" << lower2 << "\n"
             << "   Expected result:\n(  2  0  0 )\n(  0 -4  0 )\n( 21  0  0 )\n";
         throw std::runtime_error( oss.str() );
      }
   }
}
//*************************************************************************************************

} // namespace lowermatrix

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
   std::cout << "   Running LowerMatrix dense test (part 1)..." << std::endl;

   try
   {
      RUN_LOWERMATRIX_DENSE_TEST;
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during LowerMatrix dense test (part 1):\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
