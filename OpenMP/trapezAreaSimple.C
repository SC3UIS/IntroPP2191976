/*********************************
 ********TRAPEZOIDAL RULE*********
  2017 (c) Manas Sharma - https://bragitoff.com
  2023 (modified) C. Barrios       
 ********************************/
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
  double a,b,h,x,sum=0,integral,time=0.0;
  /*Ask the user for necessary input */
  clock_t inicio = clock();
  printf("\nIngresar el numero de subintervalos: ");
  scanf("%d",&n);
  printf("\nLimite inicial: ");
  scanf("%lf",&a);
  printf("\nLimite final: ");
  scanf("%lf",&b);
  /*Begin Trapezoidal Method: */
  h=fabs(b-a)/n;
  for(i=1;i<n;i++){
    x=a+i*h;
    sum=sum+f(x);
    //sirve para contar las pasadas del for
    printf("\nPasada numero: %i",i);
  }
  integral=(h/2)*(f(a)+f(b)+2*sum);
  /*Print the answer */
  printf("\nLa integral es: %lf\n",integral);
  clock_t final = clock();
  time += (double)(final - inicio) / CLOCKS_PER_SEC;
  printf("El codigo se demoro %f segundos\n",time);
}
