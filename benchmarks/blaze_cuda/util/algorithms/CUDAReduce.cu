#include <iostream>

#include <benchmark.h>

#include <blaze/Blaze.h>
#include <blaze_cuda/Blaze.h>

int main()
{
   namespace bm = benchmark;
   namespace bz = blaze;

   using v_t = bz::CUDADynamicVector<float>;
   size_t constexpr size = 1 << 22;

   v_t a(size, 10), b(size, 1);

   auto fun = [&]()
   {
      auto sum_a = bz::cuda_reduce( a, 0
         , [] BLAZE_DEVICE_CALLABLE (auto const& l, auto const& r)
         { return l + r; } );

      (void)sum_a;
   };

   for(auto i = 0U; i < 10; i++)
      std::cout << std::chrono::nanoseconds(bm::time(fun)).count() << "ns" << '\n';
}
