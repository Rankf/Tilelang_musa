//
// p8_hw_parameter_probe.mu
// 测量 S5000 硬件参数：sm_sub_partitions, warp_schedulers_per_sm,
// sfu_cores_per_sm, int32_cores_per_sm, shared_memory_throughput
//
// 编译：
//   mcc -O2 --cuda-gpu-arch=mp_31 -I/usr/local/musa/include \
//       p8_hw_parameter_probe.mu -L/usr/local/musa/lib -lmusart -o p8_probe
//
// 运行：
//   ./p8_probe <output.csv>
//

#include <musa_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>

// ============================================================
// 辅助函数
// ============================================================
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

static double mean(std::vector<double>& v) {
  double sum = 0;
  for (double x : v) sum += x;
  return sum / v.size();
}

// GPU 计时辅助
static double gpu_time_ms(musaEvent_t start, musaEvent_t stop) {
  float ms = 0;
  check(musaEventElapsedTime(&ms, start, stop), "musaEventElapsedTime");
  return (double)ms;
}

// ============================================================
// 1. sm_sub_partitions / warp_schedulers_per_sm 测量
//   通过独立 FMA 指令流测量最大指令发射率
//   每个线程使用 N 个独立累加器，允许编译器 pipeline
// ============================================================

// 独立 FP32 FMA kernel — 16 路独立 FMA，允许最大 ILP
__global__ void fma_peak16(float* out, int iterations, int dummy) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x) * 1.0001f + 1.0f;
    float x1 = x0 + 1.0f, x2 = x1 + 1.0f, x3 = x2 + 1.0f;
    float x4 = x3 + 1.0f, x5 = x4 + 1.0f, x6 = x5 + 1.0f, x7 = x6 + 1.0f;
    float x8 = x7 + 1.0f, x9 = x8 + 1.0f, x10 = x9 + 1.0f, x11 = x10 + 1.0f;
    float x12 = x11 + 1.0f, x13 = x12 + 1.0f, x14 = x13 + 1.0f, x15 = x14 + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x0 = fmaf(x0, 1.000001f, 1.0f);
        x1 = fmaf(x1, 1.000001f, 1.0f);
        x2 = fmaf(x2, 1.000001f, 1.0f);
        x3 = fmaf(x3, 1.000001f, 1.0f);
        x4 = fmaf(x4, 1.000001f, 1.0f);
        x5 = fmaf(x5, 1.000001f, 1.0f);
        x6 = fmaf(x6, 1.000001f, 1.0f);
        x7 = fmaf(x7, 1.000001f, 1.0f);
        x8 = fmaf(x8, 1.000001f, 1.0f);
        x9 = fmaf(x9, 1.000001f, 1.0f);
        x10 = fmaf(x10, 1.000001f, 1.0f);
        x11 = fmaf(x11, 1.000001f, 1.0f);
        x12 = fmaf(x12, 1.000001f, 1.0f);
        x13 = fmaf(x13, 1.000001f, 1.0f);
        x14 = fmaf(x14, 1.000001f, 1.0f);
        x15 = fmaf(x15, 1.000001f, 1.0f);
    }
    float s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 +
              x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// 4 路独立 FMA kernel — 中等 ILP
__global__ void fma_peak4(float* out, int iterations, int dummy) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x) * 1.0001f + 1.0f;
    float x1 = x0 + 1.0f, x2 = x1 + 1.0f, x3 = x2 + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x0 = fmaf(x0, 1.000001f, 1.0f);
        x1 = fmaf(x1, 1.000001f, 1.0f);
        x2 = fmaf(x2, 1.000001f, 1.0f);
        x3 = fmaf(x3, 1.000001f, 1.0f);
    }
    float s = x0 + x1 + x2 + x3;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// 1 路依赖 FMA kernel — 无 ILP (用于对比)
__global__ void fma_dep1(float* out, int iterations, int dummy) {
    float x = (float)(threadIdx.x + blockIdx.x * blockDim.x) * 1.0001f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x = fmaf(x, 1.000001f, 1.0f);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = x;
}

// 纯 INT32 计算 kernel（独立流）
__global__ void int32_peak4(int* out, int iterations, int dummy) {
    int x0 = (int)(threadIdx.x + blockIdx.x * blockDim.x) * 3 + 1;
    int x1 = x0 + 1, x2 = x1 + 1, x3 = x2 + 1;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x0 = x0 * 3 + 1;
        x1 = x1 * 3 + 1;
        x2 = x2 * 3 + 1;
        x3 = x3 * 3 + 1;
    }
    int s = x0 + x1 + x2 + x3;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// 纯 SFU 计算 kernel (exp) — 独立流
__global__ void sfu_exp_peak4(float* out, int iterations, int dummy) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x) * 0.0001f + 1.0f;
    float x1 = x0 + 1.0f, x2 = x1 + 1.0f, x3 = x2 + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x0 = expf(x0);
        x1 = expf(x1);
        x2 = expf(x2);
        x3 = expf(x3);
    }
    float s = x0 + x1 + x2 + x3;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// 纯 SFU 计算 kernel (sin) — 独立流
__global__ void sfu_sin_peak4(float* out, int iterations, int dummy) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x) * 0.0001f + 1.0f;
    float x1 = x0 + 1.0f, x2 = x1 + 1.0f, x3 = x2 + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        x0 = sinf(x0);
        x1 = sinf(x1);
        x2 = sinf(x2);
        x3 = sinf(x3);
    }
    float s = x0 + x1 + x2 + x3;
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// ============================================================
// Shared memory 带宽测量
// ============================================================
__global__ void smem_bandwidth_probe(float* out, int repeats) {
    __shared__ float smem[256];  // 1 KB per CTA
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    // 初始化 shared memory
    smem[tid] = (float)tid;
    __syncthreads();
    
    // 反复读取 shared memory (顺序访问，无 bank conflict)
    float sum = 0.0f;
    #pragma unroll 1
    for (int r = 0; r < repeats; r++) {
        sum += smem[tid];
    }
    if (tid == 0) out[bid] = sum;
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s OUTPUT_CSV\n", argv[0]);
        return 1;
    }
    
    // 获取设备属性
    musaDeviceProp prop;
    check(musaGetDeviceProperties(&prop, 0), "musaGetDeviceProperties");
    int sm_count = prop.multiProcessorCount;
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int max_threads_per_block = prop.maxThreadsPerBlock;
    
    printf("=== S5000 硬件参数测量 ===\n");
    printf("SM Count: %d\n", sm_count);
    printf("Max Threads/SM: %d\n", max_threads_per_sm);
    printf("Max Threads/Block: %d\n", max_threads_per_block);
    printf("Clock Rate: %d MHz\n", prop.clockRate);
    printf("========================\n\n");
    
    FILE* csv = fopen(argv[1], "w");
    if (!csv) { perror("fopen"); return 1; }
    
    // CSV 头
    fprintf(csv, "test,warps_per_cta,ctas_per_sm,warps_per_sm,latency_us,throughput,unit\n");
    
    // 预热
    float* d_out;
    check(musaMalloc((void**)&d_out, sizeof(float) * 1024), "musaMalloc w");
    int* d_out_i;
    check(musaMalloc((void**)&d_out_i, sizeof(int) * 1024), "musaMalloc i");
    
    // ============================================================
    // 测试 1: sm_sub_partitions — 不同 warp 数下 FP32 吞吐
    // 使用 16 路独立 FMA kernel 测量最大指令发射率
    // ============================================================
    printf("\n--- Test 1: sm_sub_partitions (FP32 peak FMA vs warps per CTA) ---\n");
    
    // 测试不同 kernel 类型 (1/4/16 路独立 FMA)
    struct FmaKernel { const char* name; const void* func; int flops_per_iter; };
    FmaKernel fma_kernels[] = {
        {"fma_dep1 (1 dep)", (const void*)fma_dep1, 2},
        {"fma_peak4 (4 indep)", (const void*)fma_peak4, 8},
        {"fma_peak16 (16 indep)", (const void*)fma_peak16, 32},
    };
    int num_fma_kernels = sizeof(fma_kernels) / sizeof(fma_kernels[0]);
    
    const int fma_iterations = 4096;
    
    for (int ki = 0; ki < num_fma_kernels; ki++) {
        printf("\n  --- %s ---\n", fma_kernels[ki].name);
        int flops_per_iter = fma_kernels[ki].flops_per_iter;
        
        int thread_configs[] = {32, 64, 128, 256, 512, 1024};
        int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);
        
        for (int ci = 0; ci < num_configs; ci++) {
            int threads = thread_configs[ci];
            int warps_per_cta = threads / 32;
            
            // 查 occupancy
            int active_blocks = 0;
            musaError_t occ_err = musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &active_blocks, fma_kernels[ki].func, threads, 0, musaOccupancyDefault);
            if (occ_err != musaSuccess || active_blocks == 0) continue;
            
            int grid = sm_count * active_blocks;
            if (grid > 65535) grid = 65535;
            
            musaEvent_t start, stop;
            check(musaEventCreate(&start), "event create");
            check(musaEventCreate(&stop), "event create");
            
            std::vector<double> times_us;
            for (int r = 0; r < 13; r++) {
                check(musaEventRecord(start, 0), "record start");
                check(musaLaunchKernel(fma_kernels[ki].func, dim3(grid), dim3(threads),
                                       (void*[]){&d_out, (void*)&fma_iterations, (void*)&r}, 0, 0),
                       "launch fma");
                check(musaEventRecord(stop, 0), "record stop");
                check(musaEventSynchronize(stop), "sync stop");
                if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
            }
            double med_us = median(times_us);
            double total_flops = (double)grid * (double)threads * (double)fma_iterations * (double)flops_per_iter;
            double tflops = total_flops / med_us / 1e6;
            
            printf("  threads=%4d (warp/CTA=%2d), CTAs/SM=%2d, warps/SM=%3d: %8.1f us, %7.2f TFLOP/s (%.1f%% peak)\n",
                   threads, warps_per_cta, active_blocks, active_blocks * warps_per_cta,
                   med_us, tflops, tflops / 23.0 * 100.0);
            fprintf(csv, "%s,%d,%d,%d,%.1f,%.2f,TFLOP/s\n",
                    fma_kernels[ki].name, warps_per_cta, active_blocks, active_blocks * warps_per_cta,
                    med_us, tflops);
            
            check(musaEventDestroy(start), "destroy start");
            check(musaEventDestroy(stop), "destroy stop");
        }
    }
    
    // ============================================================
    // 测试 2: int32_cores_per_sm — INT32 吞吐
    // ============================================================
    printf("\n\n--- Test 2: int32_cores_per_sm (INT32 peak throughput) ---\n");
    
    const int int32_iterations = 4096;
    int int32_threads = 256;
    
    int int32_active_blocks = 0;
    check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &int32_active_blocks, (const void*)int32_peak4, int32_threads, 0, musaOccupancyDefault),
        "int32 occupancy");
    printf("  INT32 kernel: threads=%d, active_blocks/SM=%d\n", int32_threads, int32_active_blocks);
    
    int int32_grid = sm_count * int32_active_blocks;
    
    {
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "event create");
        check(musaEventCreate(&stop), "event create");
        
        std::vector<double> times_us;
        for (int r = 0; r < 13; r++) {
            check(musaEventRecord(start, 0), "record start");
            check(musaLaunchKernel((const void*)int32_peak4, dim3(int32_grid), dim3(int32_threads),
                                   (void*[]){&d_out_i, (void*)&int32_iterations, (void*)&r}, 0, 0),
                   "launch int32");
            check(musaEventRecord(stop, 0), "record stop");
            check(musaEventSynchronize(stop), "sync stop");
            if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
        }
        double med_us = median(times_us);
        // 每个线程: 4 路独立 * iterations 次 * 1 op/iteration
        double total_ops = (double)int32_grid * (double)int32_threads * (double)int32_iterations * 4;
        double tops = total_ops / med_us / 1e6;
        double per_sm = tops / sm_count;
        // 理论: int32_cores_per_sm * 2 ops/cycle * 1.5 GHz
        // 实测: per_sm ops/s
        // 反推: int32_cores_per_sm ≈ per_sm / (1.5e9 * 2)
        double inferred_cores = per_sm / (1.5e9 * 2);
        printf("  INT32: %.1f us, %.2f TOP/s (%.2f GTOP/s/SM, inferred cores/SM=%.1f)\n",
               med_us, tops, per_sm / 1e9, inferred_cores);
        fprintf(csv, "int32_peak,%d,%d,%d,%.1f,%.2f,TOP/s\n",
                int32_threads/32, int32_active_blocks, int32_active_blocks * (int32_threads/32),
                med_us, tops);
        
        check(musaEventDestroy(start), "destroy start");
        check(musaEventDestroy(stop), "destroy stop");
    }
    
    // ============================================================
    // 测试 3: sfu_cores_per_sm — SFU 吞吐 (exp, sin)
    // ============================================================
    printf("\n--- Test 3: sfu_cores_per_sm (SFU throughput) ---\n");
    
    const int sfu_iterations = 2048;
    int sfu_threads = 256;
    
    struct SfuKernel { const char* name; const void* func; };
    SfuKernel sfu_kernels[] = {
        {"sfu_exp_peak4", (const void*)sfu_exp_peak4},
        {"sfu_sin_peak4", (const void*)sfu_sin_peak4},
    };
    int num_sfu_kernels = sizeof(sfu_kernels) / sizeof(sfu_kernels[0]);
    
    for (int ski = 0; ski < num_sfu_kernels; ski++) {
        int sfu_active_blocks = 0;
        check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &sfu_active_blocks, sfu_kernels[ski].func, sfu_threads, 0, musaOccupancyDefault),
            "sfu occupancy");
        printf("  %s: threads=%d, active_blocks/SM=%d\n",
               sfu_kernels[ski].name, sfu_threads, sfu_active_blocks);
        
        int sfu_grid = sm_count * sfu_active_blocks;
        
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "event create");
        check(musaEventCreate(&stop), "event create");
        
        std::vector<double> times_us;
        for (int r = 0; r < 13; r++) {
            check(musaEventRecord(start, 0), "record start");
            check(musaLaunchKernel(sfu_kernels[ski].func, dim3(sfu_grid), dim3(sfu_threads),
                                   (void*[]){&d_out, (void*)&sfu_iterations, (void*)&r}, 0, 0),
                   "launch sfu");
            check(musaEventRecord(stop, 0), "record stop");
            check(musaEventSynchronize(stop), "sync stop");
            if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
        }
        double med_us = median(times_us);
        double total_ops = (double)sfu_grid * (double)sfu_threads * (double)sfu_iterations * 4;
        double gops = total_ops / med_us / 1e3;
        double per_sm = gops / sm_count;
        // 反推 SFU cores: 假设每个 SFU core 每周期处理 1 次 SFU 操作
        // 但 SFU 操作可能有延迟，所以实际吞吐 < 核心数 * 频率
        // 如果 SFU 指令吞吐为 1/cycle/core，则 inferred_cores = per_sm / 1.5e9
        double inferred_cores = per_sm / 1.5e9;
        printf("  %s: %.1f us, %.2f Gops/s (%.2f Gops/s/SM, inferred cores/SM=%.1f)\n",
               sfu_kernels[ski].name, med_us, gops, per_sm, inferred_cores);
        fprintf(csv, "%s,%d,%d,%d,%.1f,%.2f,Gops/s\n",
                sfu_kernels[ski].name, sfu_threads/32, sfu_active_blocks,
                sfu_active_blocks * (sfu_threads/32), med_us, gops);
        
        check(musaEventDestroy(start), "destroy start");
        check(musaEventDestroy(stop), "destroy stop");
    }
    
    // ============================================================
    // 测试 4: warp_schedulers_per_sm — 固定 CTA 大小，变 grid 大小
    // 通过逐步增加 CTA/SM 数量观察吞吐拐点
    // ============================================================
    printf("\n--- Test 4: warp_schedulers_per_sm (throughput vs CTAs per SM) ---\n");
    
    // 使用 256 threads (8 warps/CTA) 的 fma_peak4 kernel
    // 逐步增加 CTAs/SM: 1, 2, 3, 4, 6, 8, 12
    int ws_threads = 256;
    int ws_warps_per_cta = ws_threads / 32;
    
    int ctas_to_test[] = {1, 2, 3, 4, 6, 8, 12};
    int num_ctas = sizeof(ctas_to_test) / sizeof(ctas_to_test[0]);
    
    printf("  Using fma_peak4 kernel, %d threads/CTA (%d warps/CTA)\n",
           ws_threads, ws_warps_per_cta);
    
    for (int ci = 0; ci < num_ctas; ci++) {
        int ctas_per_sm = ctas_to_test[ci];
        int grid = sm_count * ctas_per_sm;
        int total_warps_per_sm = ctas_per_sm * ws_warps_per_cta;
        
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "event create");
        check(musaEventCreate(&stop), "event create");
        
        std::vector<double> times_us;
        for (int r = 0; r < 13; r++) {
            check(musaEventRecord(start, 0), "record start");
            check(musaLaunchKernel((const void*)fma_peak4, dim3(grid), dim3(ws_threads),
                                   (void*[]){&d_out, (void*)&fma_iterations, (void*)&r}, 0, 0),
                   "launch ws probe");
            check(musaEventRecord(stop, 0), "record stop");
            check(musaEventSynchronize(stop), "sync stop");
            if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
        }
        double med_us = median(times_us);
        double total_flops = (double)grid * (double)ws_threads * (double)fma_iterations * 8;
        double tflops = total_flops / med_us / 1e6;
        double per_sm = tflops / sm_count;
        
        printf("  CTAs/SM=%2d, warps/SM=%3d: %8.1f us, %7.2f TFLOP/s (%.2f TFLOP/s/SM, %.1f%% peak)\n",
               ctas_per_sm, total_warps_per_sm, med_us, tflops, per_sm, tflops / 23.0 * 100.0);
        fprintf(csv, "warp_sched_sweep,%d,%d,%d,%.1f,%.2f,TFLOP/s\n",
                ws_warps_per_cta, ctas_per_sm, total_warps_per_sm, med_us, tflops);
        
        check(musaEventDestroy(start), "destroy start");
        check(musaEventDestroy(stop), "destroy stop");
    }
    
    // ============================================================
    // 测试 5: 使用 1 warp/CTA 变 grid 大小
    // ============================================================
    printf("\n--- Test 5: 1-warp CTA sweep (minimal CTA resource) ---\n");
    
    int ws1_threads = 32;  // 1 warp per CTA
    int ws1_ctas_to_test[] = {1, 2, 4, 8, 16, 32, 48, 64, 96};
    int num_ws1_ctas = sizeof(ws1_ctas_to_test) / sizeof(ws1_ctas_to_test[0]);
    
    printf("  Using fma_peak4 kernel, %d threads/CTA (1 warp/CTA)\n", ws1_threads);
    
    for (int ci = 0; ci < num_ws1_ctas; ci++) {
        int ctas_per_sm = ws1_ctas_to_test[ci];
        int grid = sm_count * ctas_per_sm;
        int total_warps_per_sm = ctas_per_sm * 1;
        
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "event create");
        check(musaEventCreate(&stop), "event create");
        
        std::vector<double> times_us;
        for (int r = 0; r < 13; r++) {
            check(musaEventRecord(start, 0), "record start");
            check(musaLaunchKernel((const void*)fma_peak4, dim3(grid), dim3(ws1_threads),
                                   (void*[]){&d_out, (void*)&fma_iterations, (void*)&r}, 0, 0),
                   "launch ws1 probe");
            check(musaEventRecord(stop, 0), "record stop");
            check(musaEventSynchronize(stop), "sync stop");
            if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
        }
        double med_us = median(times_us);
        double total_flops = (double)grid * (double)ws1_threads * (double)fma_iterations * 8;
        double tflops = total_flops / med_us / 1e6;
        double per_sm = tflops / sm_count;
        
        printf("  CTAs/SM=%2d, warps/SM=%3d: %8.1f us, %7.2f TFLOP/s (%.2f TFLOP/s/SM)\n",
               ctas_per_sm, total_warps_per_sm, med_us, tflops, per_sm);
        fprintf(csv, "warp_sched_1w,%d,%d,%d,%.1f,%.2f,TFLOP/s\n",
                1, ctas_per_sm, total_warps_per_sm, med_us, tflops);
        
        check(musaEventDestroy(start), "destroy start");
        check(musaEventDestroy(stop), "destroy stop");
    }
    printf("\n--- Test 6: Shared memory bandwidth ---\n");
    
    // 用 256 threads, 每个 CTA 256 个 float 的 shared memory
    int smem_threads = 256;
    int smem_repeats = 16384;
    
    {
        int smem_active_blocks = 0;
        // 需要查 shared memory 用量: 256 * 4 = 1024 bytes
        check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &smem_active_blocks, (const void*)smem_bandwidth_probe, smem_threads, 1024, musaOccupancyDefault),
            "smem occupancy");
        printf("  SMEM kernel: threads=%d, active_blocks/SM=%d\n", smem_threads, smem_active_blocks);
        
        int smem_grid = sm_count * smem_active_blocks;
        
        musaEvent_t start, stop;
        check(musaEventCreate(&start), "event create");
        check(musaEventCreate(&stop), "event create");
        
        std::vector<double> times_us;
        for (int r = 0; r < 13; r++) {
            check(musaEventRecord(start, 0), "record start");
            check(musaLaunchKernel((const void*)smem_bandwidth_probe, dim3(smem_grid), dim3(smem_threads),
                                   (void*[]){&d_out, (void*)&smem_repeats}, 0, 0),
                   "launch smem");
            check(musaEventRecord(stop, 0), "record stop");
            check(musaEventSynchronize(stop), "sync stop");
            if (r >= 3) times_us.push_back(gpu_time_ms(start, stop) * 1000.0);
        }
        double med_us = median(times_us);
        // 总 shared memory 读取量 = grid * threads * repeats * sizeof(float)
        double total_bytes = (double)smem_grid * (double)smem_threads * (double)smem_repeats * 4.0;
        double bw = total_bytes / med_us / 1e3;  // GB/s
        printf("  SMEM: %.1f us, %.2f GB/s (%.2f GB/s/SM, %.2f bytes/cycle/SM)\n",
               med_us, bw, bw / sm_count, bw / (sm_count * 1.5 * 1e3));
        fprintf(csv, "smem_bandwidth,%d,%d,%d,%.1f,%.2f,GB/s\n",
                smem_threads/32, smem_active_blocks, smem_active_blocks * (smem_threads/32),
                med_us, bw);
        
        check(musaEventDestroy(start), "destroy start");
        check(musaEventDestroy(stop), "destroy stop");
    }
    
    // ============================================================
    // 清理
    // ============================================================
    check(musaFree(d_out), "musaFree d_out");
    check(musaFree(d_out_i), "musaFree d_out_i");
    fclose(csv);
    
    printf("\n=== 测量完成 ===\n");
    printf("结果已保存到: %s\n", argv[1]);
    
    return 0;
}