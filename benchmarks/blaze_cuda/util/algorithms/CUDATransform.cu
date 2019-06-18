#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>

#include <iostream>
#include <type_traits>

#include <benchmark.h>

#include <blaze_cuda/Blaze.h>

namespace bm = benchmark;
namespace bz = blaze;

template<typename Exec>
void bench_case( Exec const& )
{
   bool constexpr RunsOnCPU = std::is_same_v<Exec, bm::exec::cpu>;

   using elmt_t = float;
   using nanos  = std::chrono::nanoseconds;
   using op     = bz::Pow2;

   using v_t = std::conditional_t< RunsOnCPU
                                 , bz::DynamicVector<elmt_t>
                                 , bz::CUDADynamicVector<elmt_t> >;

   for( auto i = size_t(10); i < size_t(31); i++ )
   {
      v_t a( size_t(1) << i, i );

      auto t = bm::bench_avg( [&]() {
         a = bz::map( a, op() );
         bm::no_optimize(a);
         if constexpr ( !RunsOnCPU ) cudaDeviceSynchronize();
      } );

      // Calculating results
      auto const gb   = 2 * sizeof(elmt_t) * float(a.size()) / 1000000000.f;
      auto const s    = float(nanos(t).count()) / 1000000000.f;
      auto const gb_s = gb / s;

      std::cout << "Size = 2^" << i << "; Bandwidth = " << gb_s << "GB/s\n";
   }
}


int main( int, char** ) {
   std::cout << "- GPU :\n";
   bench_case( bm::exec::gpu() );

   std::cout << "- CPU :\n";
   bench_case( bm::exec::cpu() );

   return 0;
}
