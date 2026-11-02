// tests/_phlop/app/pfm/bench_fp.c
// c[i] = a[i] * b[i] + c[i], fits in cache: 2 * N * reps flops (FMA when
// contracted). Build vectorized (-O3 -march/-mcpu=native) and scalar
// (-O2 -fno-tree-vectorize -ffp-contract=off) to compare event sets.
#include <stdio.h>
#include <stdlib.h>
#define N 4096
int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 50000;
  double *a = aligned_alloc(64, N * 8), *b = aligned_alloc(64, N * 8), *c = aligned_alloc(64, N * 8);
  for (int i = 0; i < N; ++i) a[i] = i * 1e-3, b[i] = 1.0 + i * 1e-6, c[i] = 0;
  for (int r = 0; r < reps; ++r) {
    for (int i = 0; i < N; ++i) c[i] = a[i] * b[i] + c[i];
    __asm__ volatile("" ::: "memory"); // one pass per rep, no fusing reps
  }
  printf("flops %ld result %f\n", 2L * N * reps, c[N / 2]);
  return 0;
}
