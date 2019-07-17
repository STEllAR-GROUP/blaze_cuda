#include <iostream>
#include <cstddef>

#include <blaze_cuda/Blaze.h>

int main( int, char const *[] )
{
   namespace bz = blaze;

   bz::CUDADynamicMatrix< float > cm( 10, 10, 1.f );

   cm = cm * cm + cm * cm;

   std::cout << cm << '\n';
}
