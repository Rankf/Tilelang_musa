//
// p8_l2_read_probe.mu - 只读 L2 带宽测量（简化版）
// 每个 kernel 只读不写，避免写干扰
//
#include <musa_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>

static void check(musaError_t status, const char* where) {
  if (status != musaSuccess) {
    fprintf(stderr, "ERROR %s: %d (%s)\n", where, (int)status, musaGetErrorString(status));
    exit(2);
  }
}

static double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  if (n == 0) return 0.0;
  if (n % 2 == 0) return (v[n/2 - 1] + v[n/2]) / 2.0;
  return v[n/2];
}

// 只读 kernel：每个线程读取 N 个 float
__global__ void read_only(const float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 确保每个线程读取不同的位置
    int offset = tid;
    float sum = 0.0f;
    for (int i = 0; i < N; i += 32) {
        sum += input[offset + i];
    }
    if (threadIdx.x == 0) output[blockIdx.x] = sum;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <working_set_mib> <output.csv>\n", argv[0]);
        return 1;
    }

    double ws_mib = atof(argv[1]);
    if (ws_mib <= 0) { fprintf(stderr, "invalid working set\n"); return 1; }

    musaDeviceProp prop;
    check(musaGetDeviceProperties(&prop, 0), "musaGetDeviceProperties");
    int sm_count = prop.multiProcessorCount;

    // 固定配置：256 threads, 尽可能多 CTA
    int threads = 256;
    int active_blocks = 0;
    check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &active_blocks, (const void*)read_only, threads, 0, musaOccupancyDefault), "occ");
    int grid = sm_count * active_blocks;
    int total_threads = grid * threads;

    // 工作集大小
    int64_t total_elements = (int64_t)(ws_mib * 1024 * 1024 / sizeof(float));
    int64_t alloc_elements = total_elements;

    // 每线程读取元素数
    int N = (int)(total_elements / total_threads);
    if (N < 32) N = 32;  // 最少 32 个
    // 对齐到 32
    N = (N / 32) * 32;

    // 确保缓冲区足够大
    int64_t max_offset = (int64_t)total_threads + N;
    if (max_offset > alloc_elements) alloc_elements = max_offset;

    // 分配
    float* d_in = nullptr;
    float* d_out = nullptr;
    check(musaMalloc((void**)&d_in, alloc_elements * sizeof(float)), "malloc in");
    check(musaMalloc((void**)&d_out, grid * sizeof(float)), "malloc out");

    // 用 host 初始化（避免 Memset 问题）
    std::vector<float> h_init((size_t)alloc_elements, 1.0f);
    check(musaMemcpy(d_in, h_init.data(), alloc_elements * sizeof(float), musaMemcpyHostToDevice), "memcpy init");

    // 预热
    check(musaLaunchKernel((const void*)read_only, dim3(grid), dim3(threads),
           (void*[]){&d_in, &d_out, &N}, 0, 0), "warmup");
    musaEvent_t sync;
    check(musaEventCreate(&sync), "evt");
    check(musaEventRecord(sync, 0), "rec");
    check(musaEventSynchronize(sync), "sync");

    // 计时
    std::vector<double> times_us;
    for (int r = 0; r < 13; r++) {
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "cs");
        check(musaEventCreate(&stop), "cp");
        check(musaEventRecord(start, 0), "rs");
        check(musaLaunchKernel((const void*)read_only, dim3(grid), dim3(threads),
               (void*[]){&d_in, &d_out, &N}, 0, 0), "launch");
        check(musaEventRecord(stop, 0), "rp");
        check(musaEventSynchronize(stop), "sp");
        if (r >= 3) {
            float ms = 0;
            check(musaEventElapsedTime(&ms, start, stop), "elapsed");
            times_us.push_back(ms * 1000.0);
        }
        check(musaEventDestroy(start), "ds");
        check(musaEventDestroy(stop), "dp");
    }

    double med_us = median(times_us);
    double bytes = (double)N * (double)total_threads * 4.0;
    double bw = bytes / med_us / 1e3;  // GB/s

    printf("WS=%4.0f MiB, N=%d, threads=%d, grid=%d: %.1f us, %.0f GB/s\n",
           ws_mib, N, threads, grid, med_us, bw);

    // 写 CSV
    FILE* csv = fopen(argv[2], "a");
    if (!csv) { perror("fopen"); return 1; }
    if (ftell(csv) == 0) {
        fprintf(csv, "working_set_mib,latency_us,bandwidth_gbps,notes\n");
    }
    const char* note = "L1_or_L2";
    if (ws_mib > 16.0 && ws_mib <= 60.0) note = "L2_cache";
    else if (ws_mib > 60.0) note = "DDR";
    fprintf(csv, "%.0f,%.1f,%.0f,%s\n", ws_mib, med_us, bw, note);
    fclose(csv);

    check(musaFree(d_in), "free in");
    check(musaFree(d_out), "free out");
    check(musaEventDestroy(sync), "free sync");
    return 0;
}