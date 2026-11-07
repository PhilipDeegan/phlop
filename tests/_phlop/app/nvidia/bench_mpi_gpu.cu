// tests/_phlop/app/nvidia/bench_mpi_gpu.cu
//
// load imbalance on purpose: the last rank does twice the GPU work, so the
// others wait in MPI_Allreduce. phlop.run.mpi_nsys should name it the straggler
// nvcc -O2 -lineinfo -ccbin mpicxx -o bench_mpi_gpu bench_mpi_gpu.cu
// mpirun -n 2 ./bench_mpi_gpu [steps]

#include <cstdio>
#include <cstdlib>

#include <mpi.h>
#include <nvtx3/nvToolsExt.h>

__global__ void work(int n, int reps, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = y[i];
  for (int r = 0; r < reps; ++r) v = v * 0.999f + 0.001f;
  y[i] = v;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int const steps = argc > 1 ? std::atoi(argv[1]) : 10;
  int const n = 1 << 22, reps = rank == size - 1 ? 40000 : 20000;

  float* y = nullptr;
  cudaMalloc(&y, n * sizeof(float));
  cudaMemset(y, 0, n * sizeof(float));
  for (int s = 0; s < steps; ++s) {
    nvtxRangePushA("step");
    work<<<(n + 255) / 256, 256>>>(n, reps, y);
    cudaDeviceSynchronize();
    double local = s, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nvtxRangePop();
  }
  cudaFree(y);
  if (rank == 0) std::printf("done %d ranks\n", size);
  MPI_Finalize();
  return 0;
}
