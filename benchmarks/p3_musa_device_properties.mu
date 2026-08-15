#include <musa_runtime.h>
#include <stdio.h>

int main(void) {
  musaDeviceProp prop;
  musaError_t status = musaGetDeviceProperties(&prop, 0);
  if (status != musaSuccess) {
    fprintf(stderr, "musaGetDeviceProperties failed: %d (%s)\n", (int)status,
            musaGetErrorString(status));
    return 1;
  }
  printf("name=%s\n", prop.name);
  printf("capability=%d.%d\n", prop.major, prop.minor);
  printf("multi_processor_count=%d\n", prop.multiProcessorCount);
  printf("l2_cache_size_bytes=%d\n", prop.l2CacheSize);
  printf("persisting_l2_cache_max_size_bytes=%d\n", prop.persistingL2CacheMaxSize);
  return 0;
}
