//
// p8_l2_read2_probe.mu - 修正版只读 L2 带宽测量
// 使用 coalesced + vectorized (float4) 读取
// 每个线程读取连续的一片，保证 L1/L2 cache line 完整利用
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

// 只读 kernel：coalesced float4 读取
// 每个线程读取 CHUNK 个 float4（连续块）
// 多个线程的块在内存中相邻，形成整体覆盖工作集
__global__ void read_coalesced(const float4* input, float* output, int chunks_per_thread, int chunk_stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    // 每个线程读取 chunks_per_thread 个 float4，间隔 chunk_stride
    for (int c = 0; c < chunks_per_thread; c++) {
        float4 v = input[tid + c * chunk_stride];
        sum += v.x + v.y + v.z + v.w;
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

    int threads = 256;
    int active_blocks = 0;
    check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &active_blocks, (const void*)read_coalesced, threads, 0, musaOccupancyDefault), "occ");
    int grid = sm_count * active_blocks;
    int total_threads = grid * threads;

    // 工作集大小（float4 块数）
    int64_t total_chunks = (int64_t)(ws_mib * 1024 * 1024 / (4 * sizeof(float)));
    int64_t alloc_elements = total_chunks * 4;  // float 元素数

    // 每个线程读取的 chunk 数
    int chunks_per_thread = (int)(total_chunks / total_threads);
    if (chunks_per_thread < 1) chunks_per_thread = 1;
    // 限制 chunk 数，避免太长 kernel（同时保证覆盖）
    const int MAX_CHUNKS = 256;
    if (chunks_per_thread > MAX_CHUNKS) chunks_per_thread = MAX_CHUNKS;

    // chunk_stride = total_threads，保证所有线程首轮读取覆盖整个工作集
    // 但这样线程读取的是 strided 的
    // 更好：chunk_stride = total_threads，每轮所有线程读取连续 total_threads 个 chunk
    int chunk_stride = total_threads;

    // 需要的 chunk 总数 = chunks_per_thread * chunk_stride
    int64_t needed_chunks = (int64_t)chunks_per_thread * chunk_stride;
    if (needed_chunks > alloc_elements / 4) {
        // 工作集不够，需要更大的 buffer
        // 但 buffer 大小 = 工作集，所以限制 chunks_per_thread
        chunks_per_thread = (int)((alloc_elements / 4) / chunk_stride);
        if (chunks_per_thread < 1) chunks_per_thread = 1;
    }

    // 分配
    float4* d_in = nullptr;
    float* d_out = nullptr;
    check(musaMalloc((void**)&d_in, alloc_elements * sizeof(float)), "malloc in");
    check(musaMalloc((void**)&d_out, grid * sizeof(float)), "malloc out");

    // 初始化
    std::vector<float> h_init((size_t)alloc_elements, 1.0f);
    check(musaMemcpy(d_in, h_init.data(), alloc_elements * sizeof(float), musaMemcpyHostToDevice), "memcpy init");

    // 预热
    check(musaLaunchKernel((const void*)read_coalesced, dim3(grid), dim3(threads),
           (void*[]){&d_in, &d_out, &chunks_per_thread, &chunk_stride}, 0, 0), "warmup");
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
        check(musaLaunchKernel((const void*)read_coalesced, dim3(grid), dim3(threads),
               (void*[]){&d_in, &d_out, &chunks_per_thread, &chunk_stride}, 0, 0), "launch");
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
    // 实际读取的字节数 = chunks_per_thread * total_threads * 16
    double bytes = (double)chunks_per_thread * (double)total_threads * 16.0;
    double bw = bytes / med_us / 1e3;  // GB/s

    printf("WS=%4.0f MiB, chunks/thr=%3d, stride=%d: %8.1f us, %8.0f GB/s\n",
           ws_mib, chunks_per_thread, chunk_stride, med_us, bw);

    FILE* csv = fopen(argv[2], "a");
    if (!csv) { perror("fopen"); return 1; }
    if (ftell(csv) == 0) {
        fprintf(csv, "working_set_mib,chunks_per_thread,latency_us,bandwidth_gbps,notes\n");
    }
    const char* note = "L1_or_L2";
    if (ws_mib > 16.0 && ws_mib <= 60.0) note = "L2_cache";
    else if (ws_mib > 60.0) note = "DDR";
    fprintf(csv, "%.0f,%d,%.1f,%.0f,%s\n", ws_mib, chunks_per_thread, med_us, bw, note);
    fclose(csv);

    check(musaFree(d_in), "free in");
    check(musaFree(d_out), "free out");
    check(musaEventDestroy(sync), "free sync");
    return 0;
}