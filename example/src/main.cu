#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>

#include <blaze/Blaze.h>

#include <blaze_cuda/Blaze.h>
#include <blaze_cuda/math/dense/CUDAManagedVector.h>

int main(int, char const *[])
{
   using vtype = blaze::CUDAManagedVector<float>;
   //using vtype = blaze::DynamicVector<float>;

   vtype a(1024, 10), b(1024, 10), c(1024);
   //c = a + b;

   return 0;
}
