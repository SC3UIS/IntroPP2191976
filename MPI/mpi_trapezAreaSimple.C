#include <stdio.h>
#include <math.h>
#include <mpi.h>

/* Define the function to be integrated here: */
double f(double x){
  return x*x;
}

/*Program begins*/
int main(int argc, char** argv){
  int n, i, rank, size;
  double a, b, h, x, sum = 0, integral, total_integral = 0;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    /* Ask the user for necessary input */
    printf("\nEnter the no. of sub-intervals: ");
    scanf("%d",&n);
    printf("\nEnter the initial limit: ");
    scanf("%lf",&a);
    printf("\nEnter the final limit: ");
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
  for (i = local_start; i <= local_end; i++) {
    x = a + i * h;
    sum += f(x);
  }

  /* Reduce the partial results to obtain the total sum */
  MPI_Reduce(&sum, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    /* Compute the final integral */
    integral = (h / 2) * (f(a) + f(b) + 2 * total_integral);

    /* Print the answer */
    printf("\nThe integral is: %lf\n",integral);
  }

  /* Finalize MPI */
  MPI_Finalize();

  return 0;
}
