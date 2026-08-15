// Conditional L1 set-conflict/associativity probe for S5000.

#include <musa_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(call) do {                                                     \
  musaError_t status = (call);                                               \
  if (status != musaSuccess) {                                               \
    std::fprintf(stderr, "%s failed: %s\n", #call,                         \
                 musaGetErrorString(status));                                \
    return 2;                                                                \
  }                                                                          \
} while (0)

__global__ void l1_conflict_probe(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int candidate_ways,
                                  int passes,
                                  int elements_per_cta) {
  constexpr int kActiveThreads = 32;
  constexpr int kLineElements = 16;       // 64-byte measured line candidate.
  constexpr int kSetStrideElements = 8192;  // 32 KiB capacity candidate.
  float sum = 0.0f;
  if (threadIdx.x < kActiveThreads) {
    const size_t base = static_cast<size_t>(blockIdx.x) * elements_per_cta
                        + threadIdx.x * kLineElements;
    for (int pass = 0; pass < passes; ++pass) {
      const int way = pass % candidate_ways;
      sum += __ldg(input + base + way * kSetStrideElements);
    }
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void initialize_ones(float* data, size_t elements) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) data[index] = 1.0f;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "Usage: %s CANDIDATE_WAYS PASSES SAMPLES\n", argv[0]);
    return 1;
  }
  const int ways = std::atoi(argv[1]);
  const int passes = std::atoi(argv[2]);
  const int samples = std::atoi(argv[3]);
  if (ways <= 0 || ways > 32 || passes <= ways || samples <= 0) return 1;

  musaDeviceProp prop;
  CHECK(musaGetDeviceProperties(&prop, 0));
  const int threads = 256;
  const int grid = prop.multiProcessorCount;
  const int elements_per_cta = ways * 8192 + 512;
  const size_t input_elements = static_cast<size_t>(grid) * elements_per_cta;
  const size_t output_elements = static_cast<size_t>(grid) * threads;
  float *input = nullptr, *output = nullptr;
  CHECK(musaMalloc(&input, input_elements * sizeof(float)));
  CHECK(musaMalloc(&output, output_elements * sizeof(float)));
  initialize_ones<<<static_cast<unsigned>((input_elements + 255) / 256), 256>>>(
      input, input_elements);
  CHECK(musaGetLastError());

  l1_conflict_probe<<<grid, threads>>>(input, output, ways, passes,
                                       elements_per_cta);
  CHECK(musaGetLastError());
  CHECK(musaDeviceSynchronize());
  musaEvent_t start, stop;
  CHECK(musaEventCreate(&start));
  CHECK(musaEventCreate(&stop));
  float best_ms = 1.0e30f;
  for (int sample = 0; sample < samples; ++sample) {
    CHECK(musaEventRecord(start));
    l1_conflict_probe<<<grid, threads>>>(input, output, ways, passes,
                                         elements_per_cta);
    CHECK(musaGetLastError());
    CHECK(musaEventRecord(stop));
    CHECK(musaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK(musaEventElapsedTime(&milliseconds, start, stop));
    if (milliseconds < best_ms) best_ms = milliseconds;
  }
  float observed = -1.0f;
  CHECK(musaMemcpy(&observed, output, sizeof(float), musaMemcpyDeviceToHost));
  const bool correctness = observed == static_cast<float>(passes);
  std::printf(
      "kernel=l1_conflict_probe,candidate_ways=%d,set_stride_bytes=32768,"
      "active_threads_per_cta=32,grid=%d,passes=%d,samples=%d,latency_us=%.6f,"
      "check=%.1f,expected=%.1f,correctness=%s\n",
      ways, grid, passes, samples, best_ms * 1000.0, observed,
      static_cast<float>(passes), correctness ? "pass" : "fail");
  CHECK(musaEventDestroy(start));
  CHECK(musaEventDestroy(stop));
  CHECK(musaFree(input));
  CHECK(musaFree(output));
  return correctness ? 0 : 3;
}
