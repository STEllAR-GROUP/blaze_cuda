#include <iostream>

#include <benchmark.h>

int main()
{
   namespace bm = benchmark;
   std::cout << std::chrono::nanoseconds(bm::time([](){})).count();
}
