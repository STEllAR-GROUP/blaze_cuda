//=================================================================================================
/*!
//  \file src/mathtest/dmatdmatsub/MUbMHa.cpp
//  \brief Source file for the MUbMHa dense matrix/dense matrix subtraction math test
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
#include <blaze/math/HybridMatrix.h>
#include <blaze/math/UniformMatrix.h>
#include <blazetest/mathtest/Creator.h>
#include <blazetest/mathtest/dmatdmatsub/OperationTest.h>
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
   std::cout << "   Running 'MUbMHa'..." << std::endl;

   using blazetest::mathtest::TypeA;
   using blazetest::mathtest::TypeB;

   try
   {
      // Matrix type definitions
      using MUb = blaze::UniformMatrix<TypeB>;
      using MHa = blaze::HybridMatrix<TypeA,128UL,128UL>;

      // Creator type definitions
      using CMUb = blazetest::Creator<MUb>;
      using CMHa = blazetest::Creator<MHa>;

      // Running tests with small matrices
      for( size_t i=0UL; i<=9UL; ++i ) {
         for( size_t j=0UL; j<=9UL; ++j ) {
            RUN_DMATDMATSUB_OPERATION_TEST( CMUb( i, j ), CMHa( i, j ) );
         }
      }

      // Running tests with large matrices
      RUN_DMATDMATSUB_OPERATION_TEST( CMUb(  67UL,  67UL ), CMHa(  67UL,  67UL ) );
      RUN_DMATDMATSUB_OPERATION_TEST( CMUb(  67UL, 127UL ), CMHa(  67UL, 127UL ) );
      RUN_DMATDMATSUB_OPERATION_TEST( CMUb( 128UL,  64UL ), CMHa( 128UL,  64UL ) );
      RUN_DMATDMATSUB_OPERATION_TEST( CMUb( 128UL, 128UL ), CMHa( 128UL, 128UL ) );
   }
   catch( std::exception& ex ) {
      std::cerr << "\n\n ERROR DETECTED during dense matrix/dense matrix subtraction:\n"
                << ex.what() << "\n";
      return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}
//*************************************************************************************************
