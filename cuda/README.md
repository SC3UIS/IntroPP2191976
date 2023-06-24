Entrega trabajo, grupo conformado por:
 DILAN ALESSANDRO CORREDOR DIAZ - 2191976
 Carlos Alberto Castañeda Torres - 2183073

CONCLUSIONES:

la implementación CUDA se destacó por su rendimiento computacional superior en comparación con las implementaciones OpenMP y MPI, logrando un speedup significativo. Por otro lado, la implementación MPI demostró el mayor speedup y escalabilidad, lo que la posiciona como una excelente opción para cálculos paralelos a gran escala. En cuanto a la implementación OpenMP, aunque mostró un speedup y escalabilidad moderados, demostró ser eficaz para paralelizar cálculos en un sistema de memoria compartida.

tiengo 6 salidas de 2 codigos, summaSimple y trapezAreaSimple, para cada codigo haz el analisis y comparacion de resultados, con conclusiones en formato README.md:
[daacorredord@guane03 cuda]$ ./mpi_suma Enter a positive integer: 4
Sum = 10
Tiempo de ejecucin: 0.000042 segundos
Speedup: 1.000000
Escalabilidad: 1.000000
Eficiencia: 1.000000

[daacorredord@guane03 cuda]$ ./omp_suma Enter a positive integer: 13
Sum = 91
Tiempo de ejecucin: 0.004784 segundos
Speedup: 0.000000
Escalabilidad: 0.000000
Eficiencia: 0.000000

[daacorredord@guane03 cuda]$ ./cuda_suma
Positive integer: 489
Sum = 152335
Execution Time: 0.189058 seconds
Speedup: 1.000000
Scalability: 0.003906 Efficiency: 0.003906

[daacorredord@guane03 cuda]$ ./mpi_suma Enter a positive integer: 4
Sum = 10
Tiempo de ejecucin: 0.000042 segundos
Speedup: 1.000000
Escalabilidad: 1.000000
Eficiencia: 1.000000

[daacorredord@guane03 cuda]$ ./omp_suma Enter a positive integer: 13
Sum = 91
Tiempo de ejecucin: 0.004784 segundos
Speedup: 0.000000
Escalabilidad: 0.000000
Eficiencia: 0.000000

[daacorredord@guane03 cuda]$ ./omp_trapez
Ingresar el numero de subintervalos: 5
Limite inicial: 1
Limite final: 5
Pasada numero: 3
Pasada numero: 1
Pasada numero: 4
Pasada numero: 2
La integral es: 41.760000
Tiempo transcurrido: 0.040000 segundos
Velocidad: 1044.00
Eficiencia: 65.25
Escalabilidad: 65.25
