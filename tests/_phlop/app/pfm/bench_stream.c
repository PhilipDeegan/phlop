// tests/_phlop/app/pfm/bench_stream.c
// c[i] += 2 * a[i] over two arrays far larger than any cache: each rep reads
// 2 * MB / 64 lines from RAM. usage: bench_stream [MB per array] [reps]
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
__attribute__((noinline)) void axpy(double* restrict c, double const* restrict a, long n) {
  for (long i = 0; i < n; ++i) c[i] += 2.0 * a[i];
}
int main(int argc, char** argv) {
  long mb = argc > 1 ? atol(argv[1]) : 512;
  int reps = argc > 2 ? atoi(argv[2]) : 4;
  if (mb <= 0 || mb > LONG_MAX / (1L << 20) || reps < 0) {
    fprintf(stderr, "usage: bench_stream [MB per array > 0] [reps >= 0]\n");
    return 1;
  }
  long n = mb * (1L << 20) / 8;
  double *a = aligned_alloc(64, n * 8), *c = aligned_alloc(64, n * 8);
  if (!a || !c) {
    fprintf(stderr, "bench_stream: cannot allocate 2 x %ld MB\n", mb);
    return 1;
  }
  for (long i = 0; i < n; ++i) a[i] = i, c[i] = 0;
  for (int r = 0; r < reps; ++r) {
    axpy(c, a, n);
    __asm__ volatile("" ::: "memory");
  }
  printf("expected RAM lines read %ld result %f\n", 2 * n * 8 / 64 * reps, c[n / 3]);
  return 0;
}
