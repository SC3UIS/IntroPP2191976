#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

/* Define the function to be integrated here: */
double f(double x){
  return x*x;
}

/* Program begins */
int main(){
  int n, i;
  double a, b, h, x, sum = 0, integral;
  clock_t start_time, end_time;
  double elapsed_time, sequential_time, parallel_time, speedup, efficiency, scalability;

  /* Ask the user for necessary input */
  printf("\nIngresar el numero de subintervalos: ");
  scanf("%d", &n);
  printf("\nLimite inicial: ");
  scanf("%lf", &a);
  printf("\nLimite final: ");
  scanf("%lf", &b);

  /* Begin Trapezoidal Method */
  h = fabs(b - a) / n;

  start_time = clock();
  #pragma omp parallel for private(x) reduction(+:sum)
  for(i = 1; i < n; i++){
    x = a + i * h;
    sum = sum + f(x);
    printf("\nPasada numero: %i", i);
  }
  end_time = clock();

  integral = (h / 2) * (f(a) + f(b) + 2 * sum);

  /* Print the answer */
  printf("\nLa integral es: %lf\n", integral);

  elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
  sequential_time = integral;
  parallel_time = elapsed_time;
  speedup = sequential_time / parallel_time;
  efficiency = speedup / omp_get_max_threads();
  scalability = sequential_time / (parallel_time * omp_get_max_threads());

  printf("Tiempo transcurrido: %.6f segundos\n", elapsed_time);
  printf("Velocidad: %.2f\n", speedup);
  printf("Eficiencia: %.2f\n", efficiency);
  printf("Escalabilidad: %.2f\n", scalability);

  return 0;
}
