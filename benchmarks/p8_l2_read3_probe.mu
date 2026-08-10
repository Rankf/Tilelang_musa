//
// p8_l2_read3_probe.mu - 正确的只读 L2 带宽测量
// 固定每线程读取 64 个 float4 (保证内存级并行度)
// 通过改变 stride 控制访问的数据范围（工作集）
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

// 固定 CHUNKS_PER_THREAD 次读取，变化 stride
__global__ void read_probe(const float4* in, float* out, int chunks, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    // 每个线程固定读取 chunks 个 float4，位置 = tid + i*stride
    for (int i = 0; i < chunks; i++) {
        float4 v = in[tid + i * stride];
        sum += v.x + v.y + v.z + v.w;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <output.csv>\n", argv[0]);
        return 1;
    }

    musaDeviceProp prop;
    check(musaGetDeviceProperties(&prop, 0), "musaGetDeviceProperties");
    int sm_count = prop.multiProcessorCount;
    long long l2_bytes = (long long)prop.l2CacheSize;

    printf("=== S5000 L2 read bandwidth (fixed chunks, var stride) ===\n");
    printf("SM=%d, L2=%lld MiB\n", sm_count, l2_bytes/(1024*1024));

    int threads = 256;
    int active_blocks = 0;
    check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &active_blocks, (const void*)read_probe, threads, 0, musaOccupancyDefault), "occ");
    int grid = sm_count * active_blocks;
    int total_threads = grid * threads;

    int chunks = 64;  // 固定 64 次 float4 读取/线程 = 1KB/线程

    // stride 列表（控制工作集）
    int strides[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                     49152, 65536, 98304, 131072, 196608, 262144};
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    printf("Strides to test: %d\n", num_strides);
    printf("Buffer size needed: %d KB\n", (int)(((int64_t)(63*262144 + total_threads) * 4) / 1024));

    // 最大 buffer 大小 = (chunks-1)*最大stride + total_threads
    int max_stride = strides[num_strides-1];
    int64_t max_elems = (int64_t)(chunks-1) * max_stride + total_threads;
    max_elems = (max_elems / 4) * 4;  // 对齐

    // 分配
    float4* d_in = nullptr;
    float* d_out = nullptr;
    check(musaMalloc((void**)&d_in, max_elems * sizeof(float)), "malloc in");
    check(musaMalloc((void**)&d_out, grid * sizeof(float)), "malloc out");

    std::vector<float> h_init((size_t)max_elems, 1.0f);
    check(musaMemcpy(d_in, h_init.data(), max_elems * sizeof(float), musaMemcpyHostToDevice), "memcpy init");

    FILE* csv = fopen(argv[1], "w");
    if (!csv) { perror("fopen"); return 1; }
    fprintf(csv, "stride,working_set_mib,latency_us,bandwidth_gbps,notes\n");

    for (int si = 0; si < num_strides; si++) {
        int stride = strides[si];
        // 工作集 ≈ stride * total_threads * 16 bytes (每轮覆盖 total_threads 个 float4)
        double ws_bytes = (double)stride * (double)total_threads * 16.0;
        double ws_mib = ws_bytes / (1024.0*1024.0);

        // 预热
        check(musaLaunchKernel((const void*)read_probe, dim3(grid), dim3(threads),
               (void*[]){&d_in, &d_out, &chunks, &stride}, 0, 0), "warmup");
        musaEvent_t sync;
        check(musaEventCreate(&sync), "evt");
        check(musaEventRecord(sync, 0), "rec");
        check(musaEventSynchronize(sync), "sync");
        check(musaEventDestroy(sync), "dsync");

        // 计时
        std::vector<double> times_us;
        for (int r = 0; r < 13; r++) {
            musaEvent_t start, stop;
            check(musaEventCreate(&start), "cs");
            check(musaEventCreate(&stop), "cp");
            check(musaEventRecord(start, 0), "rs");
            check(musaLaunchKernel((const void*)read_probe, dim3(grid), dim3(threads),
                   (void*[]){&d_in, &d_out, &chunks, &stride}, 0, 0), "launch");
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
        // 实际读取字节 = chunks * total_threads * 16
        double bytes = (double)chunks * (double)total_threads * 16.0;
        double bw = bytes / med_us / 1e3;  // GB/s

        const char* note = "L1/L2";
        if (ws_mib > 16.0 && ws_mib <= 60.0) note = "L2";
        else if (ws_mib > 60.0) note = "DDR";

        printf("  stride=%6d, WS=%6.1f MiB: %8.1f us, %8.0f GB/s [%s]\n",
               stride, ws_mib, med_us, bw, note);
        fprintf(csv, "%d,%.1f,%.1f,%.0f,%s\n", stride, ws_mib, med_us, bw, note);
    }

    fclose(csv);
    check(musaFree(d_in), "free in");
    check(musaFree(d_out), "free out");
    printf("=== DONE ===\n");
    return 0;
}