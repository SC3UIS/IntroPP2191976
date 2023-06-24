#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

__host__ __device__ double f(double x){
  return x*x;
}

__device__ double atomicAddDouble(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void integrate(double a, double b, int n, double h, double* result) {
  int i;
  double x, sum = 0.0;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (i = idx + 1; i < n; i += stride) {
    x = a + i * h;
    sum += f(x);
  }

  sum *= 2.0;

  atomicAddDouble(result, sum);
}

double getCurrentTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(){
  int n, blockSize, numBlocks;
  double a, b, h, integral;
  double* result;
  double* dev_result;

  printf("\nNumero de subintervalos: ");
  scanf("%d", &n);
  printf("\nLimite inicial: ");
  scanf("%lf", &a);
  printf("\nLimite final: ");
  scanf("%lf", &b);

  h = fabs(b - a) / n;

  blockSize = 256;
  numBlocks = (n + blockSize - 1) / blockSize;

  result = (double*)malloc(sizeof(double));
  cudaMalloc((void**)&dev_result, sizeof(double));

  *result = 0.0;
  cudaMemcpy(dev_result, result, sizeof(double), cudaMemcpyHostToDevice);

  double start = getCurrentTime();
  integrate<<<numBlocks, blockSize>>>(a, b, n, h, dev_result);
  cudaDeviceSynchronize();
  double end = getCurrentTime();

  cudaMemcpy(result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);

  integral = (h / 2) * (f(a) + f(b) + *result);

  printf("\nLa integral es: %lf\n", integral);

  double elapsedTime = end - start;
  printf("Tiempo transcurrido: %.6f seconds\n", elapsedTime);

  int numThreads = numBlocks * blockSize;
  double sequentialTime = integral;
  double parallelTime = elapsedTime;
  double speedup = sequentialTime / parallelTime;
  double efficiency = speedup / numThreads;
  double scalability = sequentialTime / (parallelTime * numThreads);

  printf("Velocidad: %.2f\n", speedup);
  printf("Eficiencia: %.2f\n", efficiency);
  printf("Escalabilidad: %.2f\n", scalability);

  free(result);
  }