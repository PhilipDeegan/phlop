// tests/_phlop/app/nvidia/bench_csan.cu
//
// one known bug per compute-sanitizer tool, picked by argv[1]:
//  oob     memcheck: out of bounds global write, and a leaked cudaMalloc
//  uninit  initcheck: reads device memory nothing wrote
//  race    racecheck: shared memory read/write hazard, no __syncthreads
//  sync    synccheck: __syncthreads in divergent code
//  clean   nothing wrong
// nvcc -O0 -lineinfo -o bench_csan bench_csan.cu

#include <cstdio>
#include <cstring>

__global__ void oob(float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) y[i] = 1.f;  // <= : one past the end
}

__global__ void sum(float const* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.f;
}

__global__ void race(float* y) {
  __shared__ float s[256];
  s[threadIdx.x] = threadIdx.x;
  // missing __syncthreads()
  y[threadIdx.x] = s[(threadIdx.x + 1) % blockDim.x];
}

__global__ void divergent(float* y) {
  __shared__ float s[256];
  if (threadIdx.x < 16) {
    s[threadIdx.x] = 1.f;
    __syncthreads();  // only some threads get here
  }
  y[threadIdx.x] = s[threadIdx.x % 16];
}

int main(int argc, char** argv) {
  char const* what = argc > 1 ? argv[1] : "clean";
  int const n = 1000;
  float *x = nullptr, *y = nullptr, *leak = nullptr;
  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&y, n * sizeof(float));
  cudaMemset(y, 0, n * sizeof(float));

  if (!std::strcmp(what, "oob")) {
    oob<<<4, 256>>>(y, n);
    cudaMalloc(&leak, 4096);  // never freed
  } else if (!std::strcmp(what, "uninit")) {
    sum<<<4, 256>>>(x, y, n);  // x never written
  } else if (!std::strcmp(what, "race")) {
    race<<<1, 256>>>(y);
  } else if (!std::strcmp(what, "sync")) {
    divergent<<<1, 256>>>(y);
  } else {
    cudaMemset(x, 0, n * sizeof(float));
    sum<<<4, 256>>>(x, y, n);
  }
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("%s: %s\n", what, cudaGetErrorString(e));
  cudaFree(x);
  cudaFree(y);
  return 0;
}
