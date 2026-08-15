#include <musa.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iterator>
#include <vector>

static void check(MUresult result, const char* where) {
  if (result != MUSA_SUCCESS) {
    const char* name = "unknown";
    const char* text = "unknown";
    muGetErrorName(result, &name);
    muGetErrorString(result, &text);
    fprintf(stderr, "%s failed: %d (%s: %s)\n", where, (int)result, name, text);
    exit(2);
  }
}

static int attr(MUfunction function, MUfunction_attribute attribute) {
  int value = -1;
  check(muFuncGetAttribute(&value, attribute, function), "muFuncGetAttribute");
  return value;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s MUBIN FUNCTION BLOCK_THREADS DYNAMIC_SHARED_BYTES\n", argv[0]);
    return 1;
  }
  const char* path = argv[1];
  const char* function_name = argv[2];
  const int block_threads = atoi(argv[3]);
  const size_t dynamic_shared_bytes = (size_t)strtoull(argv[4], nullptr, 10);
  std::ifstream file(path, std::ios::binary);
  if (!file) { perror(path); return 1; }
  std::vector<char> image((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  image.push_back(0);

  check(muInit(0), "muInit");
  MUdevice device;
  check(muDeviceGet(&device, 0), "muDeviceGet");
  MUcontext context;
  check(muDevicePrimaryCtxRetain(&context, device), "muDevicePrimaryCtxRetain");
  check(muCtxSetCurrent(context), "muCtxSetCurrent");
  MUmodule module;
  check(muModuleLoadData(&module, image.data()), "muModuleLoadData");
  MUfunction function;
  check(muModuleGetFunction(&function, module, function_name), "muModuleGetFunction");

  int active = -1;
  check(muOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &active, function, block_threads, dynamic_shared_bytes,
            MU_OCCUPANCY_DEFAULT),
        "muOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
  printf("function_name=%s\n", function_name);
  printf("block_threads=%d\n", block_threads);
  printf("dynamic_shared_bytes=%zu\n", dynamic_shared_bytes);
  printf("max_threads_per_block=%d\n", attr(function, MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK));
  printf("static_shared_bytes=%d\n", attr(function, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
  printf("local_bytes_per_thread=%d\n", attr(function, MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES));
  printf("registers_per_thread=%d\n", attr(function, MU_FUNC_ATTRIBUTE_NUM_REGS));
  printf("active_cta_per_mp=%d\n", active);

  check(muModuleUnload(module), "muModuleUnload");
  check(muCtxSetCurrent(nullptr), "muCtxSetCurrent null");
  check(muDevicePrimaryCtxRelease(device), "muDevicePrimaryCtxRelease");
  return 0;
}
