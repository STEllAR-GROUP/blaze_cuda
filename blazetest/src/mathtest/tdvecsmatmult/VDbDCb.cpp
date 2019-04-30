//=================================================================================================
/*!
//  \file src/mathtest/tdvecsmatmult/VDbDCb.cpp
//  \brief Source file for the VDbDCb dense vector/sparse matrix multiplication math test
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
#include <blaze/math/DiagonalMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/tdvecsmatmult/OperationTest.h>
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
   std::cout << "   Running 'VDbDCb'..." << std::endl;

   using blazetest::mathtest::TypeB;

   try
   {
      // Matrix type definitions
      using VDb = blaze::DynamicVector<TypeB>;
      using DCb = blaze::DiagonalMatrix< blaze::CompressedMatrix<TypeB> >;

      // Creator type definitions
      using CVDb = blazetest::Creator<VDb>;
      using CDCb = blazetest::Creator<DCb>;

      // Running tests with small vectors and matrices
      for( size_t i=0UL; i<=6UL; ++i ) {
         for( size_t j=0UL; j<=i; ++j ) {
            RUN_TDVECSMATMULT_OPERATION_TEST( CVDb( i ), CDCb( i, j ) );
         }
      }

      // Running tests with large vectors and matrices
      RUN_TDVECSMATMULT_OPERATION_TEST( CVDb(  67UL ), CDCb(  67UL,  7UL ) );
      RUN_TDVECSMATMULT_OPERATION_TEST( CVDb( 127UL ), CDCb( 127UL, 13UL ) );
      RUN_TDVECSMATMULT_OPERATION_TEST( CVDb(  64UL ), CDCb(  64UL,  8UL ) );
      RUN_TDVECSMATMULT_OPERATION_TEST( CVDb( 128UL ), CDCb( 128UL, 16UL ) );
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dense vector/sparse matrix multiplication:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
