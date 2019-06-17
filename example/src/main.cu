#include <iostream>
#include <cstddef>

#include <blaze/Blaze.h>

#include <blaze_cuda/Blaze.h>

int main( int, char const *[] )
{
   namespace bz = blaze;
   using std::uint32_t;

   constexpr auto exponent = 22;

   bz::CUDADynamicVector< uint32_t > cv( 1UL << exponent, 1 );

   auto val = bz::cuda_reduce( cv , uint32_t(0)
      , [] BLAZE_DEVICE_CALLABLE (auto const& l, auto const& r) { return l + r; } );

   std::cout << val << '\n' << (1UL << exponent) << '\n';
}
