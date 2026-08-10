//
// p8_int32_sfu_probe.mu - 精确测量 INT32 / SFU 吞吐
// INT32: 纯加法 (x+=1) 多 ILP 扫描
// SFU: expf / sinf 多 ILP 扫描
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

// ========== INT32 纯加法，多路独立流 ==========
// 用 i&1 使运算数据依赖，防止编译器常数折叠
__global__ void i32_add4(int* out, int iters, int seed) {
    int x0 = threadIdx.x + blockIdx.x * blockDim.x + seed;
    int x1 = x0 + 1, x2 = x1 + 1, x3 = x2 + 1;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int d = (i & 0x1) ? 1 : 2;
        x0 += d; x1 += d; x2 += d; x3 += d;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = x0 + x1 + x2 + x3;
}

__global__ void i32_add8(int* out, int iters, int seed) {
    int x0 = threadIdx.x + blockIdx.x * blockDim.x + seed;
    int x1=x0+1,x2=x1+1,x3=x2+1,x4=x3+1,x5=x4+1,x6=x5+1,x7=x6+1;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int d = (i & 0x1) ? 1 : 2;
        x0+=d;x1+=d;x2+=d;x3+=d;x4+=d;x5+=d;x6+=d;x7+=d;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = x0+x1+x2+x3+x4+x5+x6+x7;
}

__global__ void i32_add16(int* out, int iters, int seed) {
    int x0 = threadIdx.x + blockIdx.x * blockDim.x + seed;
    int x1=x0+1,x2=x1+1,x3=x2+1,x4=x3+1,x5=x4+1,x6=x5+1,x7=x6+1;
    int x8=x7+1,x9=x8+1,x10=x9+1,x11=x10+1,x12=x11+1,x13=x12+1,x14=x13+1,x15=x14+1;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        int d = (i & 0x1) ? 1 : 2;
        x0+=d;x1+=d;x2+=d;x3+=d;x4+=d;x5+=d;x6+=d;x7+=d;
        x8+=d;x9+=d;x10+=d;x11+=d;x12+=d;x13+=d;x14+=d;x15+=d;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
}

// INT32 乘加 4 路（对比基准）
__global__ void i32_mad4(int* out, int iters, int seed) {
    int x0 = threadIdx.x + blockIdx.x * blockDim.x + seed;
    int x1 = x0 + 1, x2 = x1 + 1, x3 = x2 + 1;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) { x0 = x0 * 3 + 1; x1 = x1 * 3 + 1; x2 = x2 * 3 + 1; x3 = x3 * 3 + 1; }
    if (threadIdx.x == 0) out[blockIdx.x] = x0 + x1 + x2 + x3;
}

// ========== SFU expf，多路独立流 ==========
__global__ void sfu_e1(float* out, int iters, int seed) {
    float x = (float)(threadIdx.x + blockIdx.x * blockDim.x + seed) * 0.0001f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) x = expf(x);
    if (threadIdx.x == 0) out[blockIdx.x] = x;
}

__global__ void sfu_e2(float* out, int iters, int seed) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x + seed) * 0.0001f + 1.0f;
    float x1 = x0 + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) { x0 = expf(x0); x1 = expf(x1); }
    if (threadIdx.x == 0) out[blockIdx.x] = x0 + x1;
}

__global__ void sfu_e4(float* out, int iters, int seed) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x + seed) * 0.0001f + 1.0f;
    float x1=x0+1.0f,x2=x1+1.0f,x3=x2+1.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) { x0=expf(x0);x1=expf(x1);x2=expf(x2);x3=expf(x3); }
    if (threadIdx.x == 0) out[blockIdx.x] = x0+x1+x2+x3;
}

__global__ void sfu_e8(float* out, int iters, int seed) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x + seed) * 0.0001f + 1.0f;
    float x1=x0+1,x2=x1+1,x3=x2+1,x4=x3+1,x5=x4+1,x6=x5+1,x7=x6+1;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) { x0=expf(x0);x1=expf(x1);x2=expf(x2);x3=expf(x3);x4=expf(x4);x5=expf(x5);x6=expf(x6);x7=expf(x7); }
    if (threadIdx.x == 0) out[blockIdx.x] = x0+x1+x2+x3+x4+x5+x6+x7;
}

// SFU sinf 4 路
__global__ void sfu_s4(float* out, int iters, int seed) {
    float x0 = (float)(threadIdx.x + blockIdx.x * blockDim.x + seed) * 0.0001f + 1.0f;
    float x1=x0+1.0f,x2=x1+1.0f,x3=x2+1.0f;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) { x0=sinf(x0);x1=sinf(x1);x2=sinf(x2);x3=sinf(x3); }
    if (threadIdx.x == 0) out[blockIdx.x] = x0+x1+x2+x3;
}

// ========== 辅助：单次测量 ==========
static double time_kernel(const void* func, int grid, int threads, void** args) {
    for (int w = 0; w < 3; w++) {
        check(musaLaunchKernel(func, dim3(grid), dim3(threads), args, 0, 0), "warmup");
    }
    musaEvent_t es, ep;
    check(musaEventCreate(&es), "es");
    check(musaEventCreate(&ep), "ep");
    check(musaEventRecord(es, 0), "rs0");
    check(musaEventSynchronize(es), "ss0");

    std::vector<double> times;
    for (int r = 0; r < 10; r++) {
        check(musaEventRecord(es, 0), "rs");
        check(musaLaunchKernel(func, dim3(grid), dim3(threads), args, 0, 0), "launch");
        check(musaEventRecord(ep, 0), "rp");
        check(musaEventSynchronize(ep), "sp");
        float ms = 0;
        check(musaEventElapsedTime(&ms, es, ep), "elapsed");
        times.push_back(ms * 1000.0);
    }
    check(musaEventDestroy(es), "des");
    check(musaEventDestroy(ep), "dep");
    return median(times);
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "Usage: %s OUTPUT_CSV\n", argv[0]); return 1; }
    setbuf(stdout, NULL);

    musaDeviceProp prop;
    check(musaGetDeviceProperties(&prop, 0), "musaGetDeviceProperties");
    int sm = prop.multiProcessorCount;
    printf("=== S5000 INT32 / SFU 精确测量 ===\nSM=%d\n\n", sm);

    int* d_i = nullptr; float* d_f = nullptr;
    check(musaMalloc((void**)&d_i, sm * 16 * sizeof(int)), "mi");
    check(musaMalloc((void**)&d_f, sm * 16 * sizeof(float)), "mf");

    FILE* csv = fopen(argv[1], "w");
    fprintf(csv, "test,ilp,latency_us,throughput,unit,ops_per_cycle_sm\n");

    const int TH = 256;
    // ADD 比 MAD 快，用更大迭代次数使其 compute-bound
    const int ADD_ITERS = 65536;    // 2^16
    const int MAD_ITERS = 4096;
    int seed = 42;

    // ========== INT32 ==========
    printf("--- INT32 ---\n");
    struct { const char* name; const void* f; int ilp; int ops_per_iter; int iters; } i32k[] = {
        {"i32_add4", (const void*)i32_add4, 4, 4, ADD_ITERS},
        {"i32_add8", (const void*)i32_add8, 8, 8, ADD_ITERS},
        {"i32_add16", (const void*)i32_add16, 16, 16, ADD_ITERS},
        {"i32_mad4", (const void*)i32_mad4, 4, 4, MAD_ITERS},  // 对比
    };

    for (int k = 0; k < 4; k++) {
        int active = 0;
        check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &active, i32k[k].f, TH, 0, musaOccupancyDefault), "occ");
        int grid = sm * active;
        int i = i32k[k].iters, s = seed + k;
        void* args[] = {&d_i, &i, &s};
        double us = time_kernel(i32k[k].f, grid, TH, args);
        double total_ops = (double)grid * (double)TH * (double)i32k[k].iters * (double)i32k[k].ilp;
        double tops = total_ops / us / 1e6;
        double ops_cyc = (tops / sm) / 1.5e9;
        printf("  %-12s (ILP=%2d, iters=%6d): %7.1f us, %6.2f TOP/s, %5.1f ops/cycle/SM\n",
               i32k[k].name, i32k[k].ilp, i32k[k].iters, us, tops, ops_cyc);
        fprintf(csv, "int32,%d,%d,%.1f,%.2f,TOP/s,%.1f\n", i32k[k].ilp, i32k[k].iters, us, tops, ops_cyc);
    }

    // ========== SFU expf ==========
    printf("\n--- SFU expf ---\n");
    struct { const char* name; const void* f; int ilp; } se[] = {
        {"sfu_e1", (const void*)sfu_e1, 1},
        {"sfu_e2", (const void*)sfu_e2, 2},
        {"sfu_e4", (const void*)sfu_e4, 4},
        {"sfu_e8", (const void*)sfu_e8, 8},
    };
    const int SITERS = 2048;

    for (int k = 0; k < 4; k++) {
        int active = 0;
        check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &active, se[k].f, TH, 0, musaOccupancyDefault), "occ");
        int grid = sm * active;
        int i = SITERS, s = seed + k + 10;
        void* args[] = {&d_f, &i, &s};
        double us = time_kernel(se[k].f, grid, TH, args);
        double total_ops = (double)grid * (double)TH * (double)SITERS * (double)se[k].ilp;
        double gops = total_ops / us / 1e3;
        double ops_cyc = (gops / sm) / 1.5e9;
        printf("  %-10s (ILP=%d): %7.1f us, %6.0f Gops/s, %5.2f ops/cycle/SM\n",
               se[k].name, se[k].ilp, us, gops, ops_cyc);
        fprintf(csv, "sfu_exp,%d,%.1f,%.0f,Gops/s,%.2f\n", se[k].ilp, us, gops, ops_cyc);
    }

    // ========== SFU sinf ==========
    printf("\n--- SFU sinf ---\n");
    {
        int active = 0;
        check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &active, (const void*)sfu_s4, TH, 0, musaOccupancyDefault), "occ");
        int grid = sm * active;
        int i = SITERS, s = seed + 99;
        void* args[] = {&d_f, &i, &s};
        double us = time_kernel((const void*)sfu_s4, grid, TH, args);
        double total_ops = (double)grid * (double)TH * (double)SITERS * 4.0;
        double gops = total_ops / us / 1e3;
        double ops_cyc = (gops / sm) / 1.5e9;
        printf("  sfu_s4     (ILP=4): %7.1f us, %6.0f Gops/s, %5.2f ops/cycle/SM\n", us, gops, ops_cyc);
        fprintf(csv, "sfu_sin,4,%.1f,%.0f,Gops/s,%.2f\n", us, gops, ops_cyc);
    }

    fclose(csv);
    check(musaFree(d_i), "fi");
    check(musaFree(d_f), "ff");
    printf("\n=== DONE ===\n");
    return 0;
}