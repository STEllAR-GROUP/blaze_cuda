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

   a += blaze::exp(a) * 10;

   std::cout << "val:\n" << a;
}
