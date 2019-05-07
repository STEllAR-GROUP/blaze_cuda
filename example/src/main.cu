#include <iostream>
#include <cstddef>
#include <vector>
#include <algorithm>

#include <blaze/Blaze.h>

#include <blaze_cuda/Blaze.h>
#include <blaze_cuda/math/dense/CUDACustomVector.h>
#include <blaze_cuda/math/dense/CUDADynamicVector.h>

int main(int, char const *[])
{
   std::size_t constexpr vecsize = 32;
   using elmt_type = float;

   using vtype  = blaze::CUDADynamicVector<elmt_type>;
   using cvtype = blaze::CUDACustomVector<elmt_type, false, false>;

   vtype  a_( vecsize, 10 );
   cvtype a ( a_.data(), a_.size() );

   // NB: The BLAZE_HOST_DEVICE macro is here to make the lambda
   // available on CUDA devices (if CUDA is enabled)
   a += blaze::exp( a ) * 10
      + blaze::map( a
                  , [] BLAZE_HOST_DEVICE ( auto const& n ){ return n * n; } );

   std::cout << "val:\n" << a_;
}
