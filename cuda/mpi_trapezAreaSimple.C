#include <stdio.h>
#include <math.h>
#include <mpi.h>

/* Define the function to be integrated here: */
double f(double x){
  return x*x;
}

/* Program begins */
int main(int argc, char** argv){
  int n, i, rank, size;
  double a, b, h, x, sum = 0, integral, total_integral = 0;
  double start_time, end_time, elapsed_time;
  double sequential_time, parallel_time, speedup, efficiency, scalability;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    /* Ask the user for necessary input */
    printf("\nIngresa el numero de subintervalos: ");
    scanf("%d",&n);
    printf("\nLimite inicial: ");
    scanf("%lf",&a);
    printf("\nLimite final: ");
    scanf("%lf",&b);
  }

  /* Broadcast input values to all processes */
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /* Divide the work among processes */
  h = fabs(b - a) / n;
  int local_n = n / size;
  int local_start = rank * local_n + 1;
  int local_end = (rank + 1) * local_n;
  if (rank == size - 1) {
    local_end = n - 1;
  }

  /* Perform local computation */
  start_time = MPI_Wtime();
  for (i = local_start; i <= local_end; i++) {
    x = a + i * h;
    sum += f(x);
  }
  end_time = MPI_Wtime();

  /* Reduce the partial results to obtain the total sum */
  MPI_Reduce(&sum, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    /* Compute the final integral */
    integral = (h / 2) * (f(a) + f(b) + 2 * total_integral);

    /* Print the answer */
    printf("\nLa integral es: %lf\n",integral);

    /* Calculate metrics */
    sequential_time = integral;
    parallel_time = end_time - start_time;
    speedup = sequential_time / parallel_time;
    efficiency = speedup / size;
    scalability = sequential_time / (parallel_time * size);

    printf("Tiempo transcurrido: %.6f segundos\n", parallel_time);
    printf("Velocidad: %.2f\n", speedup);
    printf("Eficiencia: %.2f\n", efficiency);
    printf("Escalabilidad: %.2f\n", scalability);
  }

  /* Finalize MPI */
  MPI_Finalize();

  return 0;
}
