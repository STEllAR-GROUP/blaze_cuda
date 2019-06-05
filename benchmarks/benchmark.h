#pragma once

#include <chrono>

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
auto time(F&& f)
{
   auto t = std::chrono::high_resolution_clock::now();
   f();
   return std::chrono::high_resolution_clock::now() - t;
}

}
