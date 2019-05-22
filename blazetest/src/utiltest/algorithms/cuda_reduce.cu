#include <blazetest/utiltest/algorithms/cuda_reduce.h>

void launch_tests()
{
   using blazetest::utiltest::cuda_reduce::launch_tests_for_type;

   launch_tests_for_type<char          >();
   launch_tests_for_type<signed char   >();
   launch_tests_for_type<unsigned char >();
   launch_tests_for_type<wchar_t       >();
   launch_tests_for_type<short         >();
   launch_tests_for_type<unsigned short>();
   launch_tests_for_type<int           >();
   launch_tests_for_type<unsigned int  >();
   launch_tests_for_type<long          >();
   launch_tests_for_type<unsigned long >();
   launch_tests_for_type<float         >();
   launch_tests_for_type<double        >();
   launch_tests_for_type<long double   >();
}

int main()
{
   launch_tests();
}
