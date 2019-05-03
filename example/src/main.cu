#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>

#include <blaze/Blaze.h>

#include <blaze_cuda/Blaze.h>
#include <blaze_cuda/math/dense/CUDADynamicVector.h>

int main(int, char const *[])
{
   std::size_t constexpr vecsize = 32;

   using vtype = blaze::CUDADynamicVector<float>;
   vtype a(vecsize, 10);

   // NB: The BLAZE_HOST_DEVICE is here to make the lambda available on CUDA devices (if CUDA is enabled)
   a += blaze::exp( a ) * 10 + blaze::map( a, [] BLAZE_HOST_DEVICE ( auto const& n ){ return n * n; } );

   std::cout << "val:\n" << a;
}
