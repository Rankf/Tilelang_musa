//
// p8_l2_bandwidth_probe.mu
// 测量 S5000 L2 缓存带宽和利用率
// 方法：只读工作集扫描，从 L1→L2→DDR 的带宽拐点
//
// 编译：
//   mcc -O2 --cuda-gpu-arch=mp_31 -I/usr/local/musa/include \
//       p8_l2_bandwidth_probe.mu -L/usr/local/musa/lib -lmusart -o p8_l2_probe
//
// 运行：
//   ./p8_l2_probe <output.csv>
//

#include <musa_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

static double gpu_time_ms(musaEvent_t start, musaEvent_t stop) {
  float ms = 0;
  check(musaEventElapsedTime(&ms, start, stop), "musaEventElapsedTime");
  return (double)ms;
}

// ============================================================
// 只读 vectorized load kernel
// 每个线程读取 N 个 float，步长为 1，累加后写一个值
// ============================================================
__global__ void read_bandwidth_probe(const float* input, float* output,
                                      int elements_per_thread) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0.0f;
    // 使用 float4 向量化读取
    for (int i = 0; i < elements_per_thread; i += 4) {
        float4 v = ((const float4*)input)[tid + i/4];
        sum += v.x + v.y + v.z + v.w;
    }
    if (threadIdx.x == 0) output[blockIdx.x] = sum;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s OUTPUT_CSV\n", argv[0]);
        return 1;
    }

    musaDeviceProp prop;
    check(musaGetDeviceProperties(&prop, 0), "musaGetDeviceProperties");
    int sm_count = prop.multiProcessorCount;
    long long l2_bytes = (long long)prop.l2CacheSize;

    printf("=== S5000 L2 带宽测量 ===\n");
    printf("SM Count: %d\n", sm_count);
    printf("L2 Cache: %lld MiB (%lld bytes)\n", l2_bytes / (1024*1024), l2_bytes);
    printf("========================\n\n");

    FILE* csv = fopen(argv[1], "w");
    if (!csv) { perror("fopen"); return 1; }
    fprintf(csv, "test,working_set_mib,latency_us,bandwidth_gbps,notes\n");

    // 配置：256 threads/CTA
    int threads = 256;
    int active_blocks = 0;
    check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &active_blocks, (const void*)read_bandwidth_probe, threads, 0, musaOccupancyDefault),
        "occupancy read");
    int grid = sm_count * active_blocks;
    printf("Kernel config: %d threads/CTA, %d CTAs/SM, grid=%d\n\n", threads, active_blocks, grid);

    // 工作集大小列表（MiB）
    double ws_sizes_mib[] = {
        1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64,
        80, 96, 128, 192, 256
    };
    int num_ws = sizeof(ws_sizes_mib) / sizeof(ws_sizes_mib[0]);

    for (int wi = 0; wi < num_ws; wi++) {
        double ws_mib = ws_sizes_mib[wi];
        long long total_elements = (long long)(ws_mib * 1024.0 * 1024.0 / sizeof(float));
        // 每线程元素数 = 总元素 / (grid * threads)
        int elems_per_thread = (int)(total_elements / (grid * threads));
        // 确保至少 4096 个元素，防止短 kernel 开销
        if (elems_per_thread < 4096) {
            elems_per_thread = 4096;
        }
        // 对齐到 4
        elems_per_thread = (elems_per_thread / 4) * 4;
        if (elems_per_thread < 4) elems_per_thread = 4;

        // 实际分配的缓冲区（至少覆盖工作集）
        long long alloc_elements = total_elements;
        // 但如果是小工作集，需要更多元素来满足 elems_per_thread
        long long needed = (long long)grid * (long long)threads + (long long)elems_per_thread / 4;
        if (needed > alloc_elements) alloc_elements = needed;

        // 分配设备内存
        float* d_input = nullptr;
        float* d_output = nullptr;
        check(musaMalloc((void**)&d_input, alloc_elements * sizeof(float)), "musaMalloc input");
        check(musaMalloc((void**)&d_output, grid * sizeof(float)), "musaMalloc output");

        // 用确定性的值初始化
        check(musaMemset(d_input, 0x3f, alloc_elements * sizeof(float)), "musaMemset");

        // 预热（让缓存 warm up）
        for (int w = 0; w < 3; w++) {
            check(musaLaunchKernel((const void*)read_bandwidth_probe,
                   dim3(grid), dim3(threads),
                   (void*[]){&d_input, &d_output, &elems_per_thread}, 0, 0),
                   "warmup launch");
        }
        // 用 event 同步代替 device 同步
        musaEvent_t warmup_done;
        check(musaEventCreate(&warmup_done), "warmup event create");
        check(musaEventRecord(warmup_done, 0), "warmup event record");
        check(musaEventSynchronize(warmup_done), "warmup event sync");
        check(musaEventDestroy(warmup_done), "warmup event destroy");

        // 正式计时
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "event create");
        check(musaEventCreate(&stop), "event create");

        const int NUM_RUNS = 10;
        std::vector<double> times_us;

        for (int r = 0; r < NUM_RUNS + 3; r++) {
            check(musaEventRecord(start, 0), "record start");
            check(musaLaunchKernel((const void*)read_bandwidth_probe,
                   dim3(grid), dim3(threads),
                   (void*[]){&d_input, &d_output, &elems_per_thread}, 0, 0),
                   "launch read");
            check(musaEventRecord(stop, 0), "record stop");
            check(musaEventSynchronize(stop), "sync stop");
            if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
        }

        double med_us = median(times_us);
        // 逻辑读取量 = elems_per_thread * grid * threads * sizeof(float)
        double logical_bytes = (double)elems_per_thread * (double)grid * (double)threads * 4.0;
        double bw = logical_bytes / med_us / 1e3;  // GB/s

        // 判断数据在哪个缓存层级
        // 实际物理读取量 ≤ 工作集大小（因为有缓存重用）
        // 但逻辑上我们计算的是总读取字节数
        char notes[64];
        if (ws_mib <= 16.0) {
            snprintf(notes, sizeof(notes), "L1_or_L2");
        } else if (ws_mib <= l2_bytes / (1024.0*1024.0)) {
            snprintf(notes, sizeof(notes), "L2_cache");
        } else {
            snprintf(notes, sizeof(notes), "DDR");
        }

        printf("  WS=%5.0f MiB (elems/thr=%5d): %8.1f us, %8.0f GB/s [%s]\n",
               ws_mib, elems_per_thread, med_us, bw, notes);
        fprintf(csv, "read_bandwidth,%.0f,%.1f,%.0f,%s\n",
                ws_mib, med_us, bw, notes);

        check(musaEventDestroy(start), "destroy start");
        check(musaEventDestroy(stop), "destroy stop");
        check(musaFree(d_input), "free input");
        check(musaFree(d_output), "free output");
    }

    fclose(csv);
    printf("\n=== 测量完成 ===\n");
    printf("结果已保存到: %s\n", argv[1]);
    return 0;
}