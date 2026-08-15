// S5000 operator-path probe for validating ordinary cached Global loads.
//
// The kernels intentionally use coalesced float4 ordinary read-only loads and
// no Shared Memory.  This makes them useful counterexamples to the TMA GEMM:
// every input byte is eligible for the ordinary L1 path and no input byte is
// TMA/L1-bypass traffic.

#include <musa_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do {                                                     \
  musaError_t status = (call);                                               \
  if (status != musaSuccess) {                                               \
    std::fprintf(stderr, "%s failed: %s\n", #call,                         \
                 musaGetErrorString(status));                                \
    return 2;                                                                \
  }                                                                          \
} while (0)

__global__ void vector_relu_probe(const float4* __restrict__ input,
                                  float4* __restrict__ output,
                                  size_t vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const float4 value = __ldg(input + index);
  output[index] = make_float4(
      fmaxf(value.x, 0.0f), fmaxf(value.y, 0.0f),
      fmaxf(value.z, 0.0f), fmaxf(value.w, 0.0f));
}

__global__ void vector_add_probe(const float4* __restrict__ left,
                                 const float4* __restrict__ right,
                                 float4* __restrict__ output,
                                 size_t vectors) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const float4 a = __ldg(left + index);
  const float4 b = __ldg(right + index);
  output[index] = make_float4(
      a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void initialize(float4* data, size_t vectors, float value) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < vectors) data[index] = make_float4(value, value, value, value);
}

static double median(std::vector<double>* values) {
  std::sort(values->begin(), values->end());
  const size_t n = values->size();
  return n % 2 ? (*values)[n / 2]
               : 0.5 * ((*values)[n / 2 - 1] + (*values)[n / 2]);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "Usage: %s relu|add ELEMENTS SAMPLES\n", argv[0]);
    return 1;
  }
  const bool is_relu = std::strcmp(argv[1], "relu") == 0;
  const bool is_add = std::strcmp(argv[1], "add") == 0;
  const size_t elements = std::strtoull(argv[2], nullptr, 10);
  const int samples = std::atoi(argv[3]);
  if ((!is_relu && !is_add) || elements == 0 || elements % 4 || samples <= 0) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  const int threads = 256;
  const size_t vectors = elements / 4;
  const int grid = static_cast<int>((vectors + threads - 1) / threads);
  float4* left = nullptr;
  float4* right = nullptr;
  float4* output = nullptr;
  CHECK(musaMalloc(&left, vectors * sizeof(float4)));
  CHECK(musaMalloc(&output, vectors * sizeof(float4)));
  if (is_add) CHECK(musaMalloc(&right, vectors * sizeof(float4)));
  initialize<<<grid, threads>>>(left, vectors, is_relu ? -1.0f : 1.0f);
  if (is_add) initialize<<<grid, threads>>>(right, vectors, 2.0f);
  CHECK(musaGetLastError());
  CHECK(musaDeviceSynchronize());

  // One warm-up stabilizes clocks and removes first-launch overhead.  MCU's
  // exact kernel-name filter excludes initialization and selects this kernel.
  if (is_relu) {
    vector_relu_probe<<<grid, threads>>>(left, output, vectors);
  } else {
    vector_add_probe<<<grid, threads>>>(left, right, output, vectors);
  }
  CHECK(musaGetLastError());
  CHECK(musaDeviceSynchronize());

  musaEvent_t start, stop;
  CHECK(musaEventCreate(&start));
  CHECK(musaEventCreate(&stop));
  std::vector<double> times_us;
  for (int sample = 0; sample < samples; ++sample) {
    CHECK(musaEventRecord(start));
    if (is_relu) {
      vector_relu_probe<<<grid, threads>>>(left, output, vectors);
    } else {
      vector_add_probe<<<grid, threads>>>(left, right, output, vectors);
    }
    CHECK(musaGetLastError());
    CHECK(musaEventRecord(stop));
    CHECK(musaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CHECK(musaEventElapsedTime(&milliseconds, start, stop));
    times_us.push_back(milliseconds * 1000.0);
  }

  float4 observed;
  CHECK(musaMemcpy(&observed, output, sizeof(float4), musaMemcpyDeviceToHost));
  const float expected = is_relu ? 0.0f : 3.0f;
  const bool correctness = observed.x == expected && observed.y == expected &&
                           observed.z == expected && observed.w == expected;
  const double latency_us = median(&times_us);
  const size_t ordinary_read_bytes = elements * sizeof(float) * (is_add ? 2 : 1);
  const size_t ordinary_write_bytes = elements * sizeof(float);
  const double effective_gb_s =
      static_cast<double>(ordinary_read_bytes + ordinary_write_bytes) /
      latency_us / 1.0e3;
  std::printf(
      "kernel=%s,op=%s,elements=%zu,grid=%d,threads=%d,samples=%d,"
      "latency_us=%.6f,ordinary_read_bytes=%zu,ordinary_write_bytes=%zu,"
      "tma_read_bytes=0,shared_bytes=0,effective_gb_s=%.6f,"
      "correctness=%s\n",
      is_relu ? "vector_relu_probe" : "vector_add_probe", argv[1], elements,
      grid, threads, samples, latency_us, ordinary_read_bytes,
      ordinary_write_bytes, effective_gb_s, correctness ? "pass" : "fail");

  CHECK(musaEventDestroy(start));
  CHECK(musaEventDestroy(stop));
  CHECK(musaFree(left));
  if (right) CHECK(musaFree(right));
  CHECK(musaFree(output));
  return correctness ? 0 : 3;
}
