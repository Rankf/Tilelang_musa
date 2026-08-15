#include <musa_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

__device__ __forceinline__ uint32_t dev_step(uint32_t x) {
  x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}
static uint32_t host_step(uint32_t x) {
  x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

template<int N>
__global__ void register_pressure(uint32_t* out, int iterations) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t values[N];
  #pragma unroll
  for (int j = 0; j < N; ++j) values[j] = index * 747796405u + (uint32_t)j * 1664525u + 1013904223u;
  for (int i = 0; i < iterations; ++i) {
    #pragma unroll
    for (int j = 0; j < N; ++j) values[j] = dev_step(values[j] ^ (uint32_t)i ^ (uint32_t)j);
  }
  uint32_t sum = 0;
  #pragma unroll
  for (int j = 0; j < N; ++j) sum += values[j];
  out[index] = sum;
}

static void check(musaError_t status, const char* where) {
  if (status != musaSuccess) {
    fprintf(stderr, "%s: %d (%s)\n", where, (int)status, musaGetErrorString(status)); exit(2);
  }
}

struct Spec { int live_values; const void* function; };

static uint32_t expected(int n, uint32_t index, int iterations) {
  std::vector<uint32_t> values((size_t)n);
  for (int j=0;j<n;++j) values[j]=index*747796405u+(uint32_t)j*1664525u+1013904223u;
  for (int i=0;i<iterations;++i) for (int j=0;j<n;++j) values[j]=host_step(values[j]^(uint32_t)i^(uint32_t)j);
  uint32_t sum=0; for (uint32_t x:values) sum+=x; return sum;
}

int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr,"usage: %s OUTPUT_CSV\n",argv[0]); return 1; }
  FILE* csv=fopen(argv[1],"w"); if(!csv){perror("fopen");return 1;}
  fprintf(csv,"live_values,block_threads,func_num_regs,func_local_bytes,func_static_shared_bytes,active_cta_per_mp,correctness\n");
  const Spec specs[]={{4,(const void*)register_pressure<4>},{16,(const void*)register_pressure<16>},{32,(const void*)register_pressure<32>},{64,(const void*)register_pressure<64>}};
  const int blocks[]={64,128,256,512};
  const int iterations=32;
  for(const Spec& spec:specs){
    musaFuncAttributes attr; check(musaFuncGetAttributes(&attr,spec.function),"musaFuncGetAttributes");
    for(int block:blocks){
      int active=-1; check(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&active,spec.function,block,0,musaOccupancyDefault),"occupancy");
      uint32_t* device=nullptr; check(musaMalloc((void**)&device,sizeof(uint32_t)*(size_t)block),"musaMalloc");
      void* args[]={&device,(void*)&iterations};
      check(musaLaunchKernel(spec.function,dim3(1),dim3((unsigned)block),args,0,0),"launch");
      std::vector<uint32_t> host((size_t)block); check(musaMemcpy(host.data(),device,sizeof(uint32_t)*(size_t)block,musaMemcpyDeviceToHost),"copy");
      bool correct=true; for(int i=0;i<block;++i) if(host[(size_t)i]!=expected(spec.live_values,(uint32_t)i,iterations)){correct=false;break;}
      fprintf(csv,"%d,%d,%d,%zu,%zu,%d,%s\n",spec.live_values,block,attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes,active,correct?"pass":"fail");
      check(musaFree(device),"musaFree"); if(!correct){fclose(csv);return 3;}
    }
  }
  fclose(csv); return 0;
}
