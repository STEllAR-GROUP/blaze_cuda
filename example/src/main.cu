#include <iostream>

#include <blaze_cuda/Blaze.h>

int main( int, char const *[] )
{
   namespace bz = blaze;

   using mat_t = bz::DynamicMatrix< float >;

   mat_t ma( 10, 5, 1.f ), mb( 5, 10, 1.f ), mc( 10, 10 );

   mc = ma * mb;

   //bz::cudaAssign( mc, ma * mb );
   //cudaDeviceSynchronize();

   std::cout << mc << '\n';
}
