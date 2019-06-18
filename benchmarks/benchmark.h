#pragma once

#include <chrono>

#if defined(__GNUC__)
#define BM_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define BM_STRONG_INLINE __forceinline
#endif


namespace benchmark {

/**
 * @brief      Timer meta-function
 *
 * @param[in]  f     Lambda to time
 *
 * @tparam     F     Type of the lambda to time
 *
 * @return     Execution time
 */
template<typename F>
inline auto time(F&& f)
{
   auto t = std::chrono::high_resolution_clock::now();
   f();
   return std::chrono::high_resolution_clock::now() - t;
}

template<typename F>
auto bench_avg(F&& f)
{
   uint32_t constexpr iter_n(20), warmup_n(5);

   // Warmup
   for(uint32_t i(0); i < warmup_n; i++) f();

   // Measurement
   decltype(time(f)) total(0);
   for( uint32_t i(0); i < iter_n; i++ ) total += time(f);
   return total / iter_n;
}

template<typename T>
inline auto no_optimize( T const& value ) {
   asm volatile( "" : : "r,m"(value) : "memory" );
}

namespace exec {
   struct gpu {};
   struct cpu {};
}

}
