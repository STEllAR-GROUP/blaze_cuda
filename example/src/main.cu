#include <iostream>

#include <blaze_cuda/Blaze.h>

int main( int, char const *[] )
{
   namespace bz = blaze;

   using mat_t = bz::CUDADynamicMatrix< float >;

   // Matrices ----------------------------------------------------------------

   mat_t ma( 10, 10, 1.f ), mb( 10, 10, 1.f ), mc( 10, 10 );

   mc = ma + mb;

   std::cout << mc << '\n';

   // Submatrices -------------------------------------------------------------

   //auto vma = bz::submatrix( ma, 0, 0, 5, 5 )
   //   , vmb = bz::submatrix( mb, 0, 0, 5, 5 )
   //   , vmc = bz::submatrix( mc, 0, 0, 5, 5 );

   mc = ma + mb;

   std::cout << mc << '\n';
}
