/*********************************
 ********TRAPEZOIDAL RULE*********
  2017 (c) Manas Sharma - https://bragitoff.com
  2023 (modified) C. Barrios       
 ********************************/
#include<stdio.h>
#include<math.h>
 
/* Define the function to be integrated here: */
double f(double x){
  return x*x;
}
 
/*Program begins*/
int main(){
  int n,i;
  double a,b,h,x,sum=0,integral;
  /*Ask the user for necessary input */
  printf("\nEnter the no. of sub-intervals: ");
  scanf("%d",&n);
  printf("\nEnter the initial limit: ");
  scanf("%lf",&a);
  printf("\nEnter the final limit: ");
  scanf("%lf",&b);
  /*Begin Trapezoidal Method: */
  h=fabs(b-a)/n;
  for(i=1;i<n;i++){
    x=a+i*h;
    sum=sum+f(x);
  }
  integral=(h/2)*(f(a)+f(b)+2*sum);
  /*Print the answer */
  printf("\nThe integral is: %lf\n",integral);
}

