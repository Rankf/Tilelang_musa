// S5000 ordinary read-only global-load probe for L1 characterization.
//
// One launch is one deterministic MCU profiling case.  Use exactly one wave
// (60 CTAs) for the capacity sweep so each MP owns one disjoint working set.

#include <musa_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                                     \
  musaError_t status = (call);                                               \
  if (status != musaSuccess) {                                               \
    std::fprintf(stderr, "%s failed: %s\n", #call,                         \
                 musaGetErrorString(status));                                \
    return 2;                                                                \
  }                                                                          \
} while (0)

__global__ void l1_cached_probe(const float* __restrict__ input,
                                float* __restrict__ output,
                                int elements_per_cta,
                                int passes) {
  const size_t base = static_cast<size_t>(blockIdx.x) * elements_per_cta;
  float sum = 0.0f;

  // All threads cooperatively sweep the complete CTA-private region.  Later
  // passes revisit exactly the same cache lines. __ldg makes the intended
  // ordinary read-only cached path explicit.
  for (int pass = 0; pass < passes; ++pass) {
    for (int i = threadIdx.x; i < elements_per_cta; i += blockDim.x) {
      sum += __ldg(input + base + i);
    }
  }

  output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void initialize_ones(float* data, size_t elements) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) data[index] = 1.0f;
}

static double median(std::vector<double>* values) {
  std::sort(values->begin(), values->end());
  const size_t n = values->size();
  return n % 2 ? (*values)[n / 2]
               : 0.5 * ((*values)[n / 2 - 1] + (*values)[n / 2]);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "Usage: %s WORKING_SET_BYTES_PER_CTA CTA_PER_MP PASSES SAMPLES\n",
                 argv[0]);
    return 1;
  }

  const size_t working_set_bytes = std::strtoull(argv[1], nullptr, 10);
  const int cta_per_mp = std::atoi(argv[2]);
  const int passes = std::atoi(argv[3]);
  const int samples = std::atoi(argv[4]);
  if (working_set_bytes < sizeof(float) || working_set_bytes % sizeof(float) ||
      cta_per_mp <= 0 || passes <= 1 || samples <= 0) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  musaDeviceProp prop;
  CHECK(musaGetDeviceProperties(&prop, 0));
  const int threads = 256;
  const int grid = prop.multiProcessorCount * cta_per_mp;
  const int elements_per_cta = static_cast<int>(working_set_bytes / sizeof(float));
  const size_t input_elements = static_cast<size_t>(grid) * elements_per_cta;
  const size_t output_elements = static_cast<size_t>(grid) * threads;

  float* input = nullptr;
  float* output = nullptr;
  CHECK(musaMalloc(&input, input_elements * sizeof(float)));
  CHECK(musaMalloc(&output, output_elements * sizeof(float)));
  initialize_ones<<<static_cast<unsigned>((input_elements + 255) / 256), 256>>>(
      input, input_elements);
  CHECK(musaGetLastError());
  CHECK(musaMemset(output, 0, output_elements * sizeof(float)));

  int occupancy = 0;
  CHECK(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &occupancy, reinterpret_cast<const void*>(l1_cached_probe), threads, 0,
      musaOccupancyDefault));

  l1_cached_probe<<<grid, threads>>>(input, output, elements_per_cta, passes);
  CHECK(musaGetLastError());
  CHECK(musaDeviceSynchronize());

  musaEvent_t start, stop;
  CHECK(musaEventCreate(&start));
  CHECK(musaEventCreate(&stop));
  std::vector<double> times_us;
  for (int sample = 0; sample < samples; ++sample) {
    CHECK(musaEventRecord(start));
    l1_cached_probe<<<grid, threads>>>(input, output, elements_per_cta, passes);
    CHECK(musaGetLastError());
    CHECK(musaEventRecord(stop));
    CHECK(musaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK(musaEventElapsedTime(&milliseconds, start, stop));
    times_us.push_back(milliseconds * 1000.0);
  }

  const double latency_us = median(&times_us);
  const double logical_bytes = static_cast<double>(working_set_bytes) * grid * passes;
  const double logical_gb_s = logical_bytes / latency_us / 1.0e3;
  float check_value = -1.0f;
  CHECK(musaMemcpy(&check_value, output, sizeof(float), musaMemcpyDeviceToHost));
  const int thread0_elements = (elements_per_cta + threads - 1) / threads;
  const float expected = static_cast<float>(thread0_elements * passes);
  const bool correctness = check_value == expected;

  std::printf(
      "kernel=l1_cached_probe,working_set_bytes_per_cta=%zu,cta_per_mp=%d,"
      "grid=%d,threads=%d,passes=%d,samples=%d,occupancy_limit=%d,"
      "latency_us=%.6f,logical_gb_s=%.6f,check=%.1f,expected=%.1f,"
      "correctness=%s\n",
      working_set_bytes, cta_per_mp, grid, threads, passes, samples, occupancy,
      latency_us, logical_gb_s, check_value, expected,
      correctness ? "pass" : "fail");

  CHECK(musaEventDestroy(start));
  CHECK(musaEventDestroy(stop));
  CHECK(musaFree(input));
  CHECK(musaFree(output));
  return correctness ? 0 : 3;
}
