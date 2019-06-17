#include <hpx/hpx_init.hpp>
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
   using op     = bz::Add;

   // Vector type: CPU or GPU
   using v_t = std::conditional_t< RunsOnCPU
                                 , bz::DynamicVector<elmt_t>
                                 , bz::CUDADynamicVector<elmt_t> >;

   // Evaluating performance on sizes from 2^10 to 2^30 (included)
   for( auto i = size_t(10); i < size_t(31); i++ )
   {
      v_t a( size_t(1) << i, i );

      auto t = bm::bench_avg( [&]() {
         if constexpr ( RunsOnCPU )
            bm::no_optimize( bz::reduce( a, op() ) );
         else
            bm::no_optimize( bz::cuda_reduce( a, elmt_t(0), op() ) );
      } );

      // Calculating results
      auto const gb   = sizeof(elmt_t) * float(a.size()) / 1000000000.f;
      auto const s    = float(nanos(t).count()) / 1000000000.f;
      auto const gb_s = gb / s;

      std::cout << "Size = 2^" << i << "; Bandwidth = " << gb_s << "GB/s\n";
   }
}

int main( int, char** )
{
   std::cout << "- GPU :\n";
   bench_case( bm::exec::gpu() );

   std::cout << "- CPU :\n";
   bench_case( bm::exec::cpu() );

   return 0;
}
