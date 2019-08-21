#include <iostream>

#include <blaze_cuda/Blaze.h>

int main( int, char const *[] )
{
   namespace bz = blaze;

   using mat_t = bz::CUDADynamicMatrix< float, blaze::rowMajor >;
   using vec_t = bz::CUDADynamicVector< float, true >;

   vec_t va( 10 ), vc( 5 );
   mat_t mb( 10, 5 );

   auto svc = bz::subvector( vc, 0, 5 );
   static_assert( !bz::IsCUDAAssignable_v< decltype(svc) > );

   //decltype(vc*mb) a("123");

   bz::cudaAssign( va, vc * mb );

   std::cout << va << '\n';
   std::cout << mb << '\n';
}
