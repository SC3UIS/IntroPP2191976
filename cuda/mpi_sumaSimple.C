#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

double get_wall_time() {
    struct timeval time;
    if (gettimeofday(&time, NULL)) {
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

int main(int argc, char** argv)
{
    int num, count, sum = 0;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Enter a positive integer: ");
        scanf("%d", &num);
    }

    double start_time = get_wall_time(); // Inicio del tiempo de ejecución

    MPI_Bcast(&num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int start = rank * (num / size) + 1;
    int end = (rank + 1) * (num / size);

    if (rank == size - 1)
    {
        end = num;
    }

    for (count = start; count <= end; ++count)
    {
        sum += count;
    }

    int totalSum = 0;
    MPI_Reduce(&sum, &totalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        double end_time = get_wall_time(); // Fin del tiempo de ejecución
        double tiempo_ejecucion = end_time - start_time;

        printf("\nSum = %d\n", totalSum);
        printf("Tiempo de ejecución: %f segundos\n", tiempo_ejecucion);

        // Cálculo de la escalabilidad, el speedup y la eficiencia
        double tiempo_secuencial = 0.0;
        MPI_Reduce(&tiempo_ejecucion, &tiempo_secuencial, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        int num_procesos = size;
        double speedup = tiempo_secuencial / tiempo_ejecucion;
        double escalabilidad = speedup / num_procesos;
        double eficiencia = speedup / num_procesos;

        printf("Speedup: %f\n", speedup);
        printf("Escalabilidad: %f\n", escalabilidad);
        printf("Eficiencia: %f\n", eficiencia);
    }

    MPI_Finalize();

    return 0;
}
