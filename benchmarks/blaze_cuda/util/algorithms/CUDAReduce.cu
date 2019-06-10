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

   using v_t = std::conditional_t< RunsOnCPU
                                 , bz::DynamicVector<elmt_t>
                                 , bz::CUDADynamicVector<elmt_t> >;

   for( auto i = size_t(10); i < size_t(31); i++ )
   {
      v_t a( size_t(1) << i, i );

      auto t = bm::bench_avg( [&]() {
         if constexpr ( RunsOnCPU )
            bm::no_optimize( bz::reduce( a, bz::Add() ) );
         else
            bm::no_optimize( bz::cuda_reduce( a, elmt_t(0), bz::Add() ) );
      } );

      auto const size_gb = float(sizeof(elmt_t)) * float(a.size()) / 1000000000.f;
      auto const t_s = float(nanos(t).count()) / 1000000000.f;
      auto const gb_s = size_gb / t_s;

      std::cout << "Size = 2^" << i << "; Bandwidth = " << gb_s << "GB/s\n";
   }
}

int main()
{
   std::cout << "- GPU :\n";
   bench_case( bm::exec::gpu() );

   std::cout << "- CPU :\n";
   bench_case( bm::exec::cpu() );
}
