#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>

#include <blaze/Blaze.h>

#include <blaze_cuda/Blaze.h>
#include <blaze_cuda/math/dense/CUDADynamicVector.h>
#include <blaze_cuda/util/algorithms/CUDATransform.h>

int main(int, char const *[])
{
   std::size_t constexpr vecsize    = 32;

   using vtype = blaze::CUDADynamicVector<float>;
   vtype a(vecsize, 10), b(vecsize, 10);

   a = a * b;

   std::cout << "a:\n" << a;
}
