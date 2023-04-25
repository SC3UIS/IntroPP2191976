/*
  Purpose:
   Codigo sencillo que calcula la integral de una funcion usando el metodo de trapecio,
    tambien cuenta la veces que se ejecuta el ciclo for y arroja el tiempo usado para ejecutar el codigo
    
   Example:
   
   31 May 2001 09:45:54 AM
  Licensing:
   This code is distributed under the GNU LGPL license.
  Modified:
    24 September 2003
  Author:
  Manas Sharma 
  OpenMP Modification:
  22 april 2023 by Dilan Corredor, Universidad Industrial de Santander dilancorr@gmail.com                   
  This OpenMP Modification makes a parallelization of the original Code...  
*/
#include<stdio.h>
#include<math.h>
#include <time.h>
/* Define the function to be integrated here: */
double f(double x){
  return x*x;
}
 
/*Program begins*/
int main(){
  int n,i;
  double a,b,h,x,sum=0,integral;
  /*Ask the user for necessary input */
  printf("\nIngresar el numero de subintervalos: ");
  scanf("%d",&n);
  printf("\nLimite inicial: ");
  scanf("%lf",&a);
  printf("\nLimite final: ");
  scanf("%lf",&b);
  /*Begin Trapezoidal Method: */
  h=fabs(b-a)/n;
  #pragma omp parallel for private(x) reduction(+:sum)
  for(i=1;i<n;i++){
    x=a+i*h;
    sum=sum+f(x);
    printf("\nPasada numero: %i",i);
  }
  integral=(h/2)*(f(a)+f(b)+2*sum);
  /*Print the answer */
  printf("\nLa integral es: %lf\n",integral);
}
