#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>

#include <blaze/Blaze.h>

#include <blaze_cuda/Blaze.h>
#include <blaze_cuda/math/dense/CUDAManagedVector.h>

int main(int, char const *[])
{
   bool        constexpr using_cuda = true;
   std::size_t constexpr vecsize    = 32;

   using vtype = std::conditional< using_cuda
                                 , blaze::CUDAManagedVector<float>
                                 , blaze::DynamicVector<float> >::type;


   vtype a(vecsize, 10), b(vecsize, 10);

   vtype c = a + b;

   cudaDeviceSynchronize();

   //[](void a){}(c);

   std::cout << c;
}
