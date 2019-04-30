//=================================================================================================
/*!
//  \file src/mathtest/smatsmatmult/SCbSCb.cpp
//  \brief Source file for the SCbSCb sparse matrix/sparse matrix multiplication math test
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
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/SymmetricMatrix.h>
#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/smatsmatmult/OperationTest.h>
#include <blazetest/system/MathTest.h>

#ifdef BLAZE_USE_HPX_THREADS
#  include <hpx/hpx_main.hpp>
#endif


//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
int main()
{
   std::cout << "   Running 'SCbSCb'..." << std::endl;

   using blazetest::mathtest::TypeB;

   try
   {
      // Matrix type definitions
      using SCb = blaze::SymmetricMatrix< blaze::CompressedMatrix<TypeB> >;

      // Creator type definitions
      using CSCb = blazetest::Creator<SCb>;

      // Running tests with small matrices
      for( size_t i=0UL; i<=6UL; ++i ) {
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i,     0UL ), CSCb( i,     0UL ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i,     0UL ), CSCb( i, 0.3*i*i ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i,     0UL ), CSCb( i,     i*i ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i, 0.3*i*i ), CSCb( i,     0UL ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i, 0.3*i*i ), CSCb( i, 0.3*i*i ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i, 0.3*i*i ), CSCb( i,     i*i ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i,     i*i ), CSCb( i,     0UL ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i,     i*i ), CSCb( i, 0.3*i*i ) );
         RUN_SMATSMATMULT_OPERATION_TEST( CSCb( i,     i*i ), CSCb( i,     i*i ) );
      }

      // Running tests with large matrices
      RUN_SMATSMATMULT_OPERATION_TEST( CSCb( 15UL,  7UL ), CSCb( 15UL,  7UL ) );
      RUN_SMATSMATMULT_OPERATION_TEST( CSCb( 37UL,  7UL ), CSCb( 37UL,  7UL ) );
      RUN_SMATSMATMULT_OPERATION_TEST( CSCb( 63UL,  7UL ), CSCb( 63UL,  7UL ) );
      RUN_SMATSMATMULT_OPERATION_TEST( CSCb( 16UL,  8UL ), CSCb( 16UL,  8UL ) );
      RUN_SMATSMATMULT_OPERATION_TEST( CSCb( 32UL,  8UL ), CSCb( 32UL,  8UL ) );
      RUN_SMATSMATMULT_OPERATION_TEST( CSCb( 64UL, 16UL ), CSCb( 64UL, 16UL ) );
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during sparse matrix/sparse matrix multiplication:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
