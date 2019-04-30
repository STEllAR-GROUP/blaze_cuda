//=================================================================================================
/*!
//  \file blazetest/mathtest/dynamicmatrix/ClassTest.h
//  \brief Header file for the DynamicMatrix class test
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

#ifndef _BLAZETEST_MATHTEST_DYNAMICMATRIX_CLASSTEST_H_
#define _BLAZETEST_MATHTEST_DYNAMICMATRIX_CLASSTEST_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <blaze/math/constraints/ColumnMajorMatrix.h>
#include <blaze/math/constraints/DenseMatrix.h>
#include <blaze/math/constraints/RequiresEvaluation.h>
#include <blaze/math/constraints/RowMajorMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/typetraits/IsRowMajorMatrix.h>
#include <blaze/util/AlignedAllocator.h>
#include <blaze/util/typetraits/AlignmentOf.h>
#include <blaze/util/constraints/SameType.h>
#include <blazetest/system/Types.h>


namespace blazetest {

namespace mathtest {

namespace dynamicmatrix {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Auxiliary class for all tests of the DynamicMatrix class template.
//
// This class represents a test suite for the blaze::DynamicMatrix class template. It performs
// a series of both compile time as well as runtime tests.
*/
class ClassTest
{
 public:
   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit ClassTest();
   // No explicitly declared copy constructor.
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

 private:
   //**Test functions******************************************************************************
   /*!\name Test functions */
   //@{
   template< typename Type >
   void testAlignment( const std::string& type );

   void testConstructors();
   void testAssignment  ();
   void testAddAssign   ();
   void testSubAssign   ();
   void testSchurAssign ();
   void testMultAssign  ();
   void testScaling     ();
   void testFunctionCall();
   void testAt          ();
   void testIterator    ();
   void testNonZeros    ();
   void testReset       ();
   void testClear       ();
   void testResize      ();
   void testExtend      ();
   void testReserve     ();
   void testShrinkToFit ();
   void testSwap        ();
   void testTranspose   ();
   void testCTranspose  ();
   void testIsDefault   ();

   template< typename Type >
   void checkRows( const Type& matrix, size_t expectedRows ) const;

   template< typename Type >
   void checkColumns( const Type& matrix, size_t expectedColumns ) const;

   template< typename Type >
   void checkCapacity( const Type& matrix, size_t minCapacity ) const;

   template< typename Type >
   void checkNonZeros( const Type& matrix, size_t expectedNonZeros ) const;

   template< typename Type >
   void checkNonZeros( const Type& matrix, size_t index, size_t expectedNonZeros ) const;
   //@}
   //**********************************************************************************************

   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   std::string test_;  //!< Label of the currently performed test.
   //@}
   //**********************************************************************************************

   //**Type definitions****************************************************************************
   using MT  = blaze::DynamicMatrix<int,blaze::rowMajor>;     //!< Type of the row-major dynamic matrix.
   using OMT = blaze::DynamicMatrix<int,blaze::columnMajor>;  //!< Type of the column-major dynamic matrix.

   using RMT  = MT::Rebind<double>::Other;   //!< Rebound row-major dynamic matrix type.
   using ORMT = OMT::Rebind<double>::Other;  //!< Rebound column-major dynamic matrix type.
   //**********************************************************************************************

   //**Compile time checks*************************************************************************
   /*! \cond BLAZE_INTERNAL */
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT                  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( MT::TransposeType   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OMT                 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( OMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RMT                 );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( RMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ORMT                );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ORMT::ResultType    );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ORMT::OppositeType  );
   BLAZE_CONSTRAINT_MUST_BE_DENSE_MATRIX_TYPE( ORMT::TransposeType );

   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( MT                  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( MT::TransposeType   );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OMT                 );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( OMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( OMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( OMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( RMT                 );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( RMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( ORMT                );
   BLAZE_CONSTRAINT_MUST_BE_COLUMN_MAJOR_MATRIX_TYPE( ORMT::ResultType    );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( ORMT::OppositeType  );
   BLAZE_CONSTRAINT_MUST_BE_ROW_MAJOR_MATRIX_TYPE   ( ORMT::TransposeType );

   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::ResultType      );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::OppositeType    );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( MT::TransposeType   );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( OMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::ResultType     );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::OppositeType   );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( RMT::TransposeType  );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ORMT::ResultType    );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ORMT::OppositeType  );
   BLAZE_CONSTRAINT_MUST_NOT_REQUIRE_EVALUATION( ORMT::TransposeType );

   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::ResultType::ElementType      );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::OppositeType::ElementType    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( MT::ElementType,   MT::TransposeType::ElementType   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( OMT::ElementType,  OMT::ResultType::ElementType     );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( OMT::ElementType,  OMT::OppositeType::ElementType   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( OMT::ElementType,  OMT::TransposeType::ElementType  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::ResultType::ElementType     );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::OppositeType::ElementType   );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( RMT::ElementType,  RMT::TransposeType::ElementType  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ORMT::ElementType, ORMT::ResultType::ElementType    );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ORMT::ElementType, ORMT::OppositeType::ElementType  );
   BLAZE_CONSTRAINT_MUST_BE_SAME_TYPE( ORMT::ElementType, ORMT::TransposeType::ElementType );
   /*! \endcond */
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Test of the alignment of different DynamicMatrix instances.
//
// \return void
// \param type The string representation of the given template type.
// \exception std::runtime_error Error detected.
//
// This function performs a test of the alignment of both a row-major and a column-major
// \f$ 7 \times 5 \f$ DynamicMatrix instance of the given element type. In case an error
// is detected, a \a std::runtime_error exception is thrown.
*/
template< typename Type >
void ClassTest::testAlignment( const std::string& type )
{
   using RowMajorMatrixType    = blaze::DynamicMatrix<Type,blaze::rowMajor>;
   using ColumnMajorMatrixType = blaze::DynamicMatrix<Type,blaze::columnMajor>;

   const size_t alignment( blaze::AlignmentOf<Type>::value );


   //=====================================================================================
   // Single matrix alignment test
   //=====================================================================================

   {
      const RowMajorMatrixType mat( 7UL, 5UL );

      const size_t rows( blaze::usePadding ? mat.rows() : 1UL );

      for( size_t i=0UL; i<rows; ++i )
      {
         const size_t deviation( reinterpret_cast<size_t>( &mat(i,0UL) ) % alignment );

         if( deviation != 0UL ) {
            std::ostringstream oss;
            oss << " Test: Single matrix alignment test (row-major)\n"
                << " Error: Invalid alignment in row " << i << " detected\n"
                << " Details:\n"
                << "   Element type      : " << type << "\n"
                << "   Expected alignment: " << alignment << "\n"
                << "   Deviation         : " << deviation << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }

   {
      const ColumnMajorMatrixType mat( 7UL, 5UL );

      const size_t columns( blaze::usePadding ? mat.columns() : 1UL );

      for( size_t j=0UL; j<columns; ++j )
      {
         const size_t deviation( reinterpret_cast<size_t>( &mat(0UL,j) ) % alignment );

         if( deviation != 0UL ) {
            std::ostringstream oss;
            oss << " Test: Single matrix alignment test (column-major)\n"
                << " Error: Invalid alignment in column " << j << " detected\n"
                << " Details:\n"
                << "   Element type      : " << type << "\n"
                << "   Expected alignment: " << alignment << "\n"
                << "   Deviation         : " << deviation << "\n";
            throw std::runtime_error( oss.str() );
         }
      }
   }


   //=====================================================================================
   // Static array alignment test
   //=====================================================================================

   {
      const RowMajorMatrixType init( 7UL, 5UL );
      const std::array<RowMajorMatrixType,7UL> mats{ init, init, init, init, init, init, init };

      for( size_t i=0UL; i<mats.size(); ++i )
      {
         const size_t rows( blaze::usePadding ? mats[i].rows() : 1UL );

         for( size_t j=0UL; j<rows; ++j )
         {
            const size_t deviation( reinterpret_cast<size_t>( &mats[i](j,0UL) ) % alignment );

            if( deviation != 0UL ) {
               std::ostringstream oss;
               oss << " Test: Static array alignment test (row-major)\n"
                   << " Error: Invalid alignment at index " << i << " in row " << j << " detected\n"
                   << " Details:\n"
                   << "   Element type      : " << type << "\n"
                   << "   Expected alignment: " << alignment << "\n"
                   << "   Deviation         : " << deviation << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }

   {
      const ColumnMajorMatrixType init( 7UL, 5UL );
      const std::array<ColumnMajorMatrixType,7UL> mats{ init, init, init, init, init, init, init };

      for( size_t i=0UL; i<mats.size(); ++i )
      {
         const size_t columns( blaze::usePadding ? mats[i].columns() : 1UL );

         for( size_t j=0UL; j<columns; ++j )
         {
            const size_t deviation( reinterpret_cast<size_t>( &mats[i](0UL,j) ) % alignment );

            if( deviation != 0UL ) {
               std::ostringstream oss;
               oss << " Test: Static array alignment test (column-major)\n"
                   << " Error: Invalid alignment at index " << i << " in column " << j << " detected\n"
                   << " Details:\n"
                   << "   Element type      : " << type << "\n"
                   << "   Expected alignment: " << alignment << "\n"
                   << "   Deviation         : " << deviation << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }


   //=====================================================================================
   // Dynamic array alignment test
   //=====================================================================================

   {
      const RowMajorMatrixType init( 7UL, 5UL );
      const std::vector<RowMajorMatrixType> mats( 7UL, init );

      for( size_t i=0UL; i<mats.size(); ++i )
      {
         const size_t rows( blaze::usePadding ? mats[i].rows() : 1UL );

         for( size_t j=0UL; j<rows; ++j )
         {
            const size_t deviation( reinterpret_cast<size_t>( &mats[i](j,0UL) ) % alignment );

            if( deviation != 0UL ) {
               std::ostringstream oss;
               oss << " Test: Dynamic array alignment test (row-major)\n"
                   << " Error: Invalid alignment at index " << i << " in row " << j << " detected\n"
                   << " Details:\n"
                   << "   Element type      : " << type << "\n"
                   << "   Expected alignment: " << alignment << "\n"
                   << "   Deviation         : " << deviation << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }

   {
      const ColumnMajorMatrixType init( 7UL, 5UL );
      const std::vector<ColumnMajorMatrixType> mats( 7UL, init );

      for( size_t i=0UL; i<mats.size(); ++i )
      {
         const size_t columns( blaze::usePadding ? mats[i].columns() : 1UL );

         for( size_t j=0UL; j<columns; ++j )
         {
            const size_t deviation( reinterpret_cast<size_t>( &mats[i](0UL,j) ) % alignment );

            if( deviation != 0UL ) {
               std::ostringstream oss;
               oss << " Test: Dynamic array alignment test (column-major)\n"
                   << " Error: Invalid alignment at index " << i << " in column " << j << " detected\n"
                   << " Details:\n"
                   << "   Element type      : " << type << "\n"
                   << "   Expected alignment: " << alignment << "\n"
                   << "   Deviation         : " << deviation << "\n";
               throw std::runtime_error( oss.str() );
            }
         }
      }
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of rows of the given dynamic matrix.
//
// \param matrix The dynamic matrix to be checked.
// \param expectedRows The expected number of rows of the dynamic matrix.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of rows of the given dynamic matrix. In case the actual number
// of rows does not correspond to the given expected number of rows, a \a std::runtime_error
// exception is thrown.
*/
template< typename Type >  // Type of the dynamic matrix
void ClassTest::checkRows( const Type& matrix, size_t expectedRows ) const
{
   if( rows( matrix ) != expectedRows ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of rows detected\n"
          << " Details:\n"
          << "   Number of rows         : " << rows( matrix ) << "\n"
          << "   Expected number of rows: " << expectedRows << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of columns of the given dynamic matrix.
//
// \param matrix The dynamic matrix to be checked.
// \param expectedRows The expected number of columns of the dynamic matrix.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of columns of the given dynamic matrix. In case the
// actual number of columns does not correspond to the given expected number of columns,
// a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic matrix
void ClassTest::checkColumns( const Type& matrix, size_t expectedColumns ) const
{
   if( columns( matrix ) != expectedColumns ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of columns detected\n"
          << " Details:\n"
          << "   Number of columns         : " << columns( matrix ) << "\n"
          << "   Expected number of columns: " << expectedColumns << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the capacity of the given dynamic matrix.
//
// \param matrix The dynamic matrix to be checked.
// \param minCapacity The expected minimum capacity of the dynamic matrix.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the capacity of the given dynamic matrix. In case the actual capacity
// is smaller than the given expected minimum capacity, a \a std::runtime_error exception is
// thrown.
*/
template< typename Type >  // Type of the dynamic matrix
void ClassTest::checkCapacity( const Type& matrix, size_t minCapacity ) const
{
   if( capacity( matrix ) < minCapacity ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Capacity                 : " << capacity( matrix ) << "\n"
          << "   Expected minimum capacity: " << minCapacity << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements of the given dynamic matrix.
//
// \param matrix The dynamic matrix to be checked.
// \param expectedNonZeros The expected number of non-zero elements of the dynamic matrix.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements of the given dynamic matrix. In
// case the actual number of non-zero elements does not correspond to the given expected
// number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic matrix
void ClassTest::checkNonZeros( const Type& matrix, size_t expectedNonZeros ) const
{
   if( nonZeros( matrix ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( matrix ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( matrix ) < nonZeros( matrix ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( matrix ) << "\n"
          << "   Capacity           : " << capacity( matrix ) << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Checking the number of non-zero elements in a specific row/column of the given dynamic matrix.
//
// \param matrix The dynamic matrix to be checked.
// \param index The row/column to be checked.
// \param expectedNonZeros The expected number of non-zero elements in the specified row/column.
// \return void
// \exception std::runtime_error Error detected.
//
// This function checks the number of non-zero elements in the specified row/column of the
// given dynamic matrix. In case the actual number of non-zero elements does not correspond
// to the given expected number, a \a std::runtime_error exception is thrown.
*/
template< typename Type >  // Type of the dynamic matrix
void ClassTest::checkNonZeros( const Type& matrix, size_t index, size_t expectedNonZeros ) const
{
   if( nonZeros( matrix, index ) != expectedNonZeros ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid number of non-zero elements in "
          << ( blaze::IsRowMajorMatrix<Type>::value ? "row " : "column " ) << index << "\n"
          << " Details:\n"
          << "   Number of non-zeros         : " << nonZeros( matrix, index ) << "\n"
          << "   Expected number of non-zeros: " << expectedNonZeros << "\n";
      throw std::runtime_error( oss.str() );
   }

   if( capacity( matrix, index ) < nonZeros( matrix, index ) ) {
      std::ostringstream oss;
      oss << " Test: " << test_ << "\n"
          << " Error: Invalid capacity detected in "
          << ( blaze::IsRowMajorMatrix<Type>::value ? "row " : "column " ) << index << "\n"
          << " Details:\n"
          << "   Number of non-zeros: " << nonZeros( matrix, index ) << "\n"
          << "   Capacity           : " << capacity( matrix, index ) << "\n";
      throw std::runtime_error( oss.str() );
   }
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL TEST FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Testing the functionality of the DynamicMatrix class template.
//
// \return void
*/
void runTest()
{
   ClassTest();
}
//*************************************************************************************************




//=================================================================================================
//
//  MACRO DEFINITIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Macro for the execution of the DynamicMatrix class test.
*/
#define RUN_DYNAMICMATRIX_CLASS_TEST \
   blazetest::mathtest::dynamicmatrix::runTest()
/*! \endcond */
//*************************************************************************************************

} // namespace dynamicmatrix

} // namespace mathtest

} // namespace blazetest

#endif
