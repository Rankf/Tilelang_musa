#include <musa_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

static uint32_t step_host(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__device__ __forceinline__ uint32_t step_device(uint32_t x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

__global__ void p4_s0(uint32_t* out, int iterations) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = index * 747796405u + 2891336453u;
  for (int i = 0; i < iterations; ++i) x = step_device(x ^ (uint32_t)i);
  out[index] = x;
}

#define DECL_SHARED_KERNEL(NAME, WORDS) \
__global__ void NAME(uint32_t* out, int iterations) { \
  __shared__ uint32_t pad[WORDS]; \
  const uint32_t tid = threadIdx.x; \
  const uint32_t index = blockIdx.x * blockDim.x + tid; \
  for (uint32_t p = tid; p < WORDS; p += blockDim.x) pad[p] = p * 1664525u + 1013904223u; \
  __syncthreads(); \
  const uint32_t mask = pad[(tid * 17u) & ((WORDS) - 1u)]; \
  uint32_t x = index * 747796405u + 2891336453u; \
  for (int i = 0; i < iterations; ++i) x = step_device(x ^ mask ^ (uint32_t)i); \
  out[index] = x; \
}

DECL_SHARED_KERNEL(p4_s8k, 2048)
DECL_SHARED_KERNEL(p4_s16k, 4096)
DECL_SHARED_KERNEL(p4_s32k, 8192)

struct KernelSpec {
  const char* id;
  const void* func;
  int requested_shared_bytes;
};

static void fail(musaError_t status, const char* where) {
  if (status != musaSuccess) {
    fprintf(stderr, "%s: %d (%s)\n", where, (int)status, musaGetErrorString(status));
    exit(2);
  }
}

static double percentile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  const double pos = (values.size() - 1) * q;
  const size_t lo = (size_t)pos;
  const size_t hi = std::min(lo + 1, values.size() - 1);
  return values[lo] + (values[hi] - values[lo]) * (pos - lo);
}

static int query_active_blocks(const KernelSpec& spec, int block_threads) {
  int active = -1;
  // MUSA 4.3.5 exports the WithFlags ABI symbol; the convenience no-flags
  // spelling is declared by the headers but is not exported by libmusart.
  fail(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
           &active, spec.func, block_threads, 0, musaOccupancyDefault),
       "musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  return active;
}

static void check_output(uint32_t* device_out, int count, int iterations, int block_threads,
                         int shared_bytes) {
  std::vector<uint32_t> host((size_t)count);
  fail(musaMemcpy(host.data(), device_out, sizeof(uint32_t) * (size_t)count,
                  musaMemcpyDeviceToHost), "musaMemcpy output");
  const uint32_t words = (uint32_t)(shared_bytes / 4);
  for (int index = 0; index < count; ++index) {
    const uint32_t tid = (uint32_t)(index % block_threads);
    uint32_t mask = 0;
    if (words) {
      const uint32_t slot = (tid * 17u) & (words - 1u);
      mask = slot * 1664525u + 1013904223u;
    }
    uint32_t x = (uint32_t)index * 747796405u + 2891336453u;
    for (int i = 0; i < iterations; ++i) x = step_host(x ^ mask ^ (uint32_t)i);
    if (host[(size_t)index] != x) {
      fprintf(stderr, "correctness mismatch index=%d got=%u expected=%u\n", index,
              host[(size_t)index], x);
      exit(3);
    }
  }
}

static void emit_resource(FILE* csv, const KernelSpec& spec, int block_threads,
                          int multiprocessors) {
  musaFuncAttributes attr;
  fail(musaFuncGetAttributes(&attr, spec.func), "musaFuncGetAttributes");
  const int active = query_active_blocks(spec, block_threads);
  fprintf(csv,
          "resource,%s,%d,%d,0,0,%d,%d,%d,%zu,%d,not_timed,,,,,\n",
          spec.id, spec.requested_shared_bytes, block_threads, multiprocessors, active,
          attr.numRegs, attr.sharedSizeBytes, attr.maxThreadsPerBlock);
}

static void emit_timed(FILE* csv, const char* kind, const KernelSpec& spec,
                       int block_threads, int grid_blocks, int iterations, int samples) {
  const int count = grid_blocks * block_threads;
  uint32_t* out = NULL;
  fail(musaMalloc((void**)&out, sizeof(uint32_t) * (size_t)count), "musaMalloc output");
  void* args[] = {&out, &iterations};
  for (int i = 0; i < 3; ++i) {
    fail(musaLaunchKernel(spec.func, dim3((unsigned)grid_blocks), dim3((unsigned)block_threads),
                          args, 0, (musaStream_t)0), "warmup launch");
  }
  fail(musaDeviceSynchronize(), "warmup synchronize");
  check_output(out, count, iterations, block_threads, spec.requested_shared_bytes);
  std::vector<double> times;
  for (int i = 0; i < samples; ++i) {
    musaEvent_t start, end;
    fail(musaEventCreate(&start), "musaEventCreate start");
    fail(musaEventCreate(&end), "musaEventCreate end");
    fail(musaEventRecord(start, 0), "musaEventRecord start");
    fail(musaLaunchKernel(spec.func, dim3((unsigned)grid_blocks), dim3((unsigned)block_threads),
                          args, 0, (musaStream_t)0), "timed launch");
    fail(musaEventRecord(end, 0), "musaEventRecord end");
    fail(musaEventSynchronize(end), "musaEventSynchronize end");
    float elapsed_ms = 0.0f;
    fail(musaEventElapsedTime(&elapsed_ms, start, end), "musaEventElapsedTime");
    times.push_back((double)elapsed_ms * 1000.0);
    musaEventDestroy(start);
    musaEventDestroy(end);
  }
  musaFuncAttributes attr;
  fail(musaFuncGetAttributes(&attr, spec.func), "musaFuncGetAttributes timed");
  const int active = query_active_blocks(spec, block_threads);
  const double median_us = percentile(times, 0.5);
  const double p10_us = percentile(times, 0.1);
  const double p90_us = percentile(times, 0.9);
  const double rate = (double)count * (double)iterations / (median_us * 1e-6);
  fprintf(csv,
          "%s,%s,%d,%d,%d,%d,,%d,%d,%zu,%d,pass,%.6f,%.6f,%.6f,%.3f,\n",
          kind, spec.id, spec.requested_shared_bytes, block_threads, grid_blocks, iterations,
          active, attr.numRegs, attr.sharedSizeBytes, attr.maxThreadsPerBlock, median_us,
          p10_us, p90_us, rate);
  fail(musaFree(out), "musaFree output");
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT_CSV\n", argv[0]);
    return 1;
  }
  FILE* csv = fopen(argv[1], "w");
  if (!csv) { perror("fopen"); return 1; }
  fprintf(csv, "kind,kernel_id,requested_static_shared_bytes,block_threads,grid_blocks,iterations,multiprocessors,api_active_blocks_per_mp,func_num_regs,func_static_shared_bytes,func_max_threads_per_block,correctness,event_median_us,event_p10_us,event_p90_us,thread_iterations_per_s,error\n");
  musaDeviceProp prop;
  fail(musaGetDeviceProperties(&prop, 0), "musaGetDeviceProperties");
  fprintf(csv, "device_properties,device0,0,0,0,0,%d,%d,%d,%zu,%d,not_applicable,,,,,name=%s;shared_per_mp=%zu;regs_per_mp=%d;max_blocks_per_mp=%d;warp=%d\n",
          prop.multiProcessorCount, prop.maxBlocksPerMultiProcessor, prop.regsPerMultiprocessor,
          prop.sharedMemPerMultiprocessor, prop.maxThreadsPerBlock, prop.name,
          prop.sharedMemPerMultiprocessor, prop.regsPerMultiprocessor,
          prop.maxBlocksPerMultiProcessor, prop.warpSize);
  const KernelSpec specs[] = {
      {"s0", (const void*)p4_s0, 0},
      {"s8k", (const void*)p4_s8k, 8192},
      {"s16k", (const void*)p4_s16k, 16384},
      {"s32k", (const void*)p4_s32k, 32768},
  };
  const int blocks[] = {64, 128, 256, 384, 512};
  for (const KernelSpec& spec : specs)
    for (int block : blocks) emit_resource(csv, spec, block, prop.multiProcessorCount);

  const int iterations = 512;
  const int samples = 15;
  const KernelSpec& s0 = specs[0];
  for (int block : blocks) {
    const int active = query_active_blocks(s0, block);
    emit_timed(csv, "block_sweep", s0, block,
               prop.multiProcessorCount * active * 4, iterations, samples);
  }
  for (int spec_index = 1; spec_index < 4; ++spec_index) {
    const int active = query_active_blocks(specs[spec_index], 256);
    emit_timed(csv, "shared_sweep", specs[spec_index], 256,
               prop.multiProcessorCount * active * 4, iterations, samples);
  }
  const int active = query_active_blocks(s0, 256);
  const int capacity = prop.multiProcessorCount * active;
  const int wave_grids[] = {1, capacity - 1, capacity, capacity + 1,
                            2 * capacity - 1, 2 * capacity, 2 * capacity + 1};
  for (int grid : wave_grids)
    emit_timed(csv, "wave_sweep", s0, 256, grid, iterations, samples);
  fclose(csv);
  return 0;
}
