// tests/_phlop/app/nvidia/bench_gpu.cu
//
// small CUDA program with known things for the phlop nvidia reports to find:
//  - "setup" NVTX range: H2D copies, pageable then pinned
//  - "compute" NVTX range: a fast saxpy and a slow, uncoalesced strided kernel
//  - "idle" NVTX range: the host sleeps, the GPU has nothing to do
//  - "teardown": D2H copy
// nvcc -O2 -lineinfo -o bench_gpu bench_gpu.cu
// argv[1]: iterations (default 20)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <nvtx3/nvToolsExt.h>

#define CHECK(x)                                                                \
  do {                                                                          \
    cudaError_t e = (x);                                                        \
    if (e != cudaSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__global__ void saxpy(int n, float a, float const* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

// each thread walks memory with a large stride: uncoalesced, memory bound
__global__ void strided(int n, int stride, float const* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = 0;
  for (int k = 0; k < 32; ++k) acc += x[(static_cast<long>(i) * stride + k * 977) % n];
  y[i] = acc;
}

int main(int argc, char** argv) {
  int const iters = argc > 1 ? std::atoi(argv[1]) : 20;
  int const n     = 1 << 24;
  size_t const bytes = n * sizeof(float);

  nvtxRangePushA("setup");
  std::vector<float> h(n, 1.f);
  float* pinned = nullptr;
  CHECK(cudaMallocHost(&pinned, bytes));
  for (int i = 0; i < n; ++i) pinned[i] = 2.f;
  float *x = nullptr, *y = nullptr;
  CHECK(cudaMalloc(&x, bytes));
  CHECK(cudaMalloc(&y, bytes));
  CHECK(cudaMemcpy(x, h.data(), bytes, cudaMemcpyHostToDevice));  // pageable
  CHECK(cudaMemcpy(y, pinned, bytes, cudaMemcpyHostToDevice));     // pinned
  nvtxRangePop();

  nvtxRangePushA("compute");
  int const threads = 256, blocks = (n + threads - 1) / threads;
  for (int it = 0; it < iters; ++it) {
    saxpy<<<blocks, threads>>>(n, 2.f, x, y);
    strided<<<blocks, threads>>>(n, 4099, x, y);
  }
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  nvtxRangePushA("idle");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  nvtxRangePop();

  nvtxRangePushA("teardown");
  CHECK(cudaMemcpy(h.data(), y, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaFree(x));
  CHECK(cudaFree(y));
  CHECK(cudaFreeHost(pinned));
  nvtxRangePop();

  std::printf("y[0] %f\n", h[0]);
  return 0;
}
