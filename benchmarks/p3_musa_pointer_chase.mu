#include <musa_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(call) do { \
  musaError_t status = (call); \
  if (status != musaSuccess) { \
    std::fprintf(stderr, "%s failed: %s\n", #call, musaGetErrorString(status)); \
    return 1; \
  } \
} while (0)

// One dependent global load per iteration. volatile prevents load elimination.
__global__ void pointer_chase(const int* __restrict__ next, int* out, int iterations) {
  int index = 0;
  volatile const int* dependent_next = next;
  for (int i = 0; i < iterations; ++i) {
    index = dependent_next[index];
  }
  out[0] = index;
}

int main() {
  const size_t bytes_list[] = {4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864};
  const int iterations = 262144;
  const int samples = 7;
  std::printf("workset_bytes,iterations,sample_index,latency_us,ns_per_dependent_load,final_index\n");
  for (size_t bytes : bytes_list) {
    const int n = static_cast<int>(bytes / sizeof(int));
    std::vector<int> host(n);
    for (int i = 0; i < n; ++i) host[i] = i;
    std::mt19937 rng(static_cast<unsigned>(20260729 + bytes));
    std::shuffle(host.begin(), host.end(), rng);
    std::vector<int> next(n);
    for (int i = 0; i < n; ++i) next[host[i]] = host[(i + 1) % n];
    int *device_next = nullptr, *device_out = nullptr, final_index = -1;
    CHECK(musaMalloc(&device_next, bytes));
    CHECK(musaMalloc(&device_out, sizeof(int)));
    CHECK(musaMemcpy(device_next, next.data(), bytes, musaMemcpyHostToDevice));
    pointer_chase<<<1, 1>>>(device_next, device_out, iterations);
    CHECK(musaGetLastError());
    CHECK(musaDeviceSynchronize());
    for (int sample = 0; sample < samples; ++sample) {
      musaEvent_t start, end;
      CHECK(musaEventCreate(&start));
      CHECK(musaEventCreate(&end));
      CHECK(musaEventRecord(start));
      pointer_chase<<<1, 1>>>(device_next, device_out, iterations);
      CHECK(musaGetLastError());
      CHECK(musaEventRecord(end));
      CHECK(musaEventSynchronize(end));
      float milliseconds = 0.0f;
      CHECK(musaEventElapsedTime(&milliseconds, start, end));
      CHECK(musaMemcpy(&final_index, device_out, sizeof(int), musaMemcpyDeviceToHost));
      std::printf("%zu,%d,%d,%.9f,%.9f,%d\n", bytes, iterations, sample,
                  milliseconds * 1000.0f, milliseconds * 1.0e6f / iterations, final_index);
      CHECK(musaEventDestroy(start));
      CHECK(musaEventDestroy(end));
    }
    CHECK(musaFree(device_next));
    CHECK(musaFree(device_out));
  }
  return 0;
}
