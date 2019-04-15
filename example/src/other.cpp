#include <blaze/Blaze.h>

blaze::HasSIMDEnabled<float> v;

#ifdef __CUDACC__
   #define BLAZE_HOST_DEVICE __host__ __device__
#else
   #define BLAZE_HOST_DEVICE
#endif

void BLAZE_HOST_DEVICE fff() {}
