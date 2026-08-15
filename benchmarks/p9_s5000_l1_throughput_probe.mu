// S5000 L1-resident ordinary-load throughput probe.

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

__global__ void l1_throughput_probe(const float4* __restrict__ input,
                                    float* __restrict__ output,
                                    int passes,
                                    int vectors_per_cta) {
  const size_t base = static_cast<size_t>(blockIdx.x) * vectors_per_cta;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  // A changing, coalesced index prevents loop-invariant load hoisting while
  // cycling over the same 16 KiB CTA-private cache-resident region.
  for (int pass = 0; pass < passes; ++pass) {
    const int index = (threadIdx.x + pass * 257) & (vectors_per_cta - 1);
    const float4 v0 = __ldg(input + base + index);
    const float4 v1 = __ldg(input + base + ((index + 257) & (vectors_per_cta - 1)));
    const float4 v2 = __ldg(input + base + ((index + 514) & (vectors_per_cta - 1)));
    const float4 v3 = __ldg(input + base + ((index + 771) & (vectors_per_cta - 1)));
    a0 += v0.x + v0.y + v0.z + v0.w;
    a1 += v1.x + v1.y + v1.z + v1.w;
    a2 += v2.x + v2.y + v2.z + v2.w;
    a3 += v3.x + v3.y + v3.z + v3.w;
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void initialize_ones(float4* data, size_t vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < vectors) data[index] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "Usage: %s CTA_PER_MP PASSES SAMPLES\n", argv[0]);
    return 1;
  }
  const int cta_per_mp = std::atoi(argv[1]);
  const int passes = std::atoi(argv[2]);
  const int samples = std::atoi(argv[3]);
  if (cta_per_mp <= 0 || passes <= 0 || samples <= 0) return 1;

  musaDeviceProp prop;
  CHECK(musaGetDeviceProperties(&prop, 0));
  const int threads = 256;
  const int grid = prop.multiProcessorCount * cta_per_mp;
  const int vectors_per_cta = 1024;  // 16 KiB per CTA, power of two.
  const size_t input_vectors = static_cast<size_t>(grid) * vectors_per_cta;
  const size_t output_elements = static_cast<size_t>(grid) * threads;
  float4* input = nullptr;
  float* output = nullptr;
  CHECK(musaMalloc(&input, input_vectors * sizeof(float4)));
  CHECK(musaMalloc(&output, output_elements * sizeof(float)));
  initialize_ones<<<static_cast<unsigned>((input_vectors + 255) / 256), 256>>>(
      input, input_vectors);
  CHECK(musaGetLastError());

  l1_throughput_probe<<<grid, threads>>>(input, output, passes, vectors_per_cta);
  CHECK(musaGetLastError());
  CHECK(musaDeviceSynchronize());

  musaEvent_t start, stop;
  CHECK(musaEventCreate(&start));
  CHECK(musaEventCreate(&stop));
  float best_ms = 1.0e30f;
  for (int sample = 0; sample < samples; ++sample) {
    CHECK(musaEventRecord(start));
    l1_throughput_probe<<<grid, threads>>>(input, output, passes, vectors_per_cta);
    CHECK(musaGetLastError());
    CHECK(musaEventRecord(stop));
    CHECK(musaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK(musaEventElapsedTime(&milliseconds, start, stop));
    if (milliseconds < best_ms) best_ms = milliseconds;
  }

  float observed = -1.0f;
  CHECK(musaMemcpy(&observed, output, sizeof(float), musaMemcpyDeviceToHost));
  const float expected = static_cast<float>(passes * 16);
  const bool correctness = observed == expected;
  const double latency_us = best_ms * 1000.0;
  const double logical_bytes = static_cast<double>(grid) * threads * passes * 64;
  const double logical_gb_s = logical_bytes / latency_us / 1e3;
  std::printf(
      "kernel=l1_throughput_probe,working_set_bytes_per_cta=16384,"
      "cta_per_mp=%d,grid=%d,threads=%d,passes=%d,samples=%d,"
      "latency_us=%.6f,logical_gb_s=%.6f,check=%.1f,expected=%.1f,"
      "correctness=%s\n",
      cta_per_mp, grid, threads, passes, samples, latency_us, logical_gb_s,
      observed, expected, correctness ? "pass" : "fail");
  CHECK(musaEventDestroy(start));
  CHECK(musaEventDestroy(stop));
  CHECK(musaFree(input));
  CHECK(musaFree(output));
  return correctness ? 0 : 3;
}
