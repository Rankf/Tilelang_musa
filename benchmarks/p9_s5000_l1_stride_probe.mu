// S5000 ordinary read-only load stride/coalescing probe.

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

__global__ void l1_stride_probe(const float* __restrict__ input,
                                float* __restrict__ output,
                                int stride_elements,
                                int passes,
                                int footprint_elements) {
  const size_t base = static_cast<size_t>(blockIdx.x) * footprint_elements;
  const int index = threadIdx.x * stride_elements;
  float sum = 0.0f;
  for (int pass = 0; pass < passes; ++pass) {
    sum += __ldg(input + base + index);
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
    std::fprintf(stderr, "Usage: %s STRIDE_BYTES CTA_PER_MP PASSES SAMPLES\n",
                 argv[0]);
    return 1;
  }
  const int stride_bytes = std::atoi(argv[1]);
  const int cta_per_mp = std::atoi(argv[2]);
  const int passes = std::atoi(argv[3]);
  const int samples = std::atoi(argv[4]);
  if (stride_bytes < 4 || stride_bytes % 4 || cta_per_mp <= 0 ||
      passes <= 1 || samples <= 0) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  musaDeviceProp prop;
  CHECK(musaGetDeviceProperties(&prop, 0));
  const int threads = 256;
  const int grid = prop.multiProcessorCount * cta_per_mp;
  const int stride_elements = stride_bytes / sizeof(float);
  const int footprint_elements = threads * stride_elements;
  const size_t input_elements = static_cast<size_t>(grid) * footprint_elements;
  const size_t output_elements = static_cast<size_t>(grid) * threads;
  float *input = nullptr, *output = nullptr;
  CHECK(musaMalloc(&input, input_elements * sizeof(float)));
  CHECK(musaMalloc(&output, output_elements * sizeof(float)));
  initialize_ones<<<static_cast<unsigned>((input_elements + 255) / 256), 256>>>(
      input, input_elements);
  CHECK(musaGetLastError());
  CHECK(musaMemset(output, 0, output_elements * sizeof(float)));

  l1_stride_probe<<<grid, threads>>>(input, output, stride_elements, passes,
                                     footprint_elements);
  CHECK(musaGetLastError());
  CHECK(musaDeviceSynchronize());

  musaEvent_t start, stop;
  CHECK(musaEventCreate(&start));
  CHECK(musaEventCreate(&stop));
  std::vector<double> times_us;
  for (int sample = 0; sample < samples; ++sample) {
    CHECK(musaEventRecord(start));
    l1_stride_probe<<<grid, threads>>>(input, output, stride_elements, passes,
                                       footprint_elements);
    CHECK(musaGetLastError());
    CHECK(musaEventRecord(stop));
    CHECK(musaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK(musaEventElapsedTime(&milliseconds, start, stop));
    times_us.push_back(milliseconds * 1000.0);
  }

  const double latency_us = median(&times_us);
  const double requested_bytes = static_cast<double>(grid) * threads * passes * 4;
  const double requested_gb_s = requested_bytes / latency_us / 1e3;
  float observed = -1.0f;
  CHECK(musaMemcpy(&observed, output, sizeof(float), musaMemcpyDeviceToHost));
  const bool correctness = observed == static_cast<float>(passes);
  std::printf(
      "kernel=l1_stride_probe,stride_bytes=%d,footprint_span_bytes_per_cta=%d,"
      "cta_per_mp=%d,grid=%d,threads=%d,passes=%d,samples=%d,"
      "latency_us=%.6f,requested_gb_s=%.6f,check=%.1f,expected=%.1f,"
      "correctness=%s\n",
      stride_bytes, footprint_elements * 4, cta_per_mp, grid, threads, passes,
      samples, latency_us, requested_gb_s, observed, static_cast<float>(passes),
      correctness ? "pass" : "fail");

  CHECK(musaEventDestroy(start));
  CHECK(musaEventDestroy(stop));
  CHECK(musaFree(input));
  CHECK(musaFree(output));
  return correctness ? 0 : 3;
}
