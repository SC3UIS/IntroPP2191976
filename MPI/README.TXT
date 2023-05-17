En esta archivo podemos ver los tiempos de ejecucion que tomo cada uno de estos tres programas omp_trapezAreaSimple.C, trapezAreaSimple.C y mpi_trapezAreaSimple.C;
 viendo claramente como el mpi_trapezAreaSimple.C es el que menos tiempo toma en ejecutarse. tiene el menor tiempo, seguido de trapezAreaSimple.C y por ultimo omp_trapezAreaSimple.C


##  `omp_trapezAreaSimple.C`
time ./omp_trapecio

Enter the no. of sub-intervals: 5

Enter the initial limit: 4

Enter the final limit: 12

The integral is: 558.080000

real    0m23.119s
user    0m0.000s
sys     0m0.009s

## `trapezAreaSimple.C`
time ./trapezArea

Enter the no. of sub-intervals: 5

Enter the initial limit: 4

Enter the final limit: 12

The integral is: 558.080000

real    0m5.027s
user    0m0.000s
sys     0m0.002s

## `mpi_trapezAreaSimple.C` 
time mpirun -np 12 ./mpi_trapez

Enter the no. of sub-intervals: 5

Enter the initial limit: 4

Enter the final limit: 12

The integral is: 558.080000

real    0m3.266s
user    0m32.799s
sys     0m2.445s