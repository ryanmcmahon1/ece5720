/* 
 * Row by column matrix multiply
 * In a loop increase matrix dimension n from MIN_SIZE doubling
 * on each pass until MAX_SIZE is reached
 *
 * Timing results are stored in file rbyc.csv in a linear array
 *
 * to compile: gcc -std=gnu99 -O3 -o rm756_hw1_2 rm756_hw1_2.c -lm
 * run code: ./netid_hw1_2
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// set range for testing
# define MIN_SIZE    1<<6
# define MAX_SIZE    1<<10
# define BILLION     1000000000L

int main(int argc, char **argv){

  // loop and other indices
  int i = 0, j = 0, k = 0, n = 0;

  // other parameters
  float diff = 0;
  float arr[5];

  // open file to record time measurements
  FILE *fp = NULL;
  fp = fopen("rbyc.csv", "w");

  // declare matrices
  float **a, **b, **c;

  // time measurement variables
  double time;
  struct timespec start, end;

  // get clock resolution
  clock_getres(CLOCK_MONOTONIC, &start);
  printf("resolution of CLOCK_MONOTONIC is %ld ns\n", start.tv_nsec);

  // if using random matrices, set seed srand48(1);
  n = MAX_SIZE;

  // for a check of correctness use special matrices
  // then set matrices to what is needed

  // allocate memory and initialize a 
  a = (float **) malloc(n * sizeof(float *));
  for(i = 0; i < n; i++) {
      a[i] = (float *) malloc(n * sizeof(float));
      for(j = 0; j < n; j++) {
          a[i][j] = 1.0*i;
      }
  }
  // allocate memory and initialize b 
  b = (float **) malloc(n * sizeof(float *));
  for(i = 0; i < n; i++) {
      b[i] = (float *) malloc(n * sizeof(float));
      for(j = 0; j < n; j++) {
          b[i][j] = 1.0*j;
      }
  }

  // allocate memory for c 
  c = (float **) malloc(n * sizeof(float *));
  for(i = 0; i < n; i++) {
      c[i] = (float *) malloc(n * sizeof(float));
      for(j = 0; j < n; j++) {
          c[i][j] = 0;
      }
  }

  // ------ loop from MIN_SIZE, doubling the size, up to MAX_SIZE -----
  for (n = MIN_SIZE; n <= MAX_SIZE; n = n * 2) {
    
    // start clock
    clock_gettime(CLOCK_MONOTONIC, &start);

    // looping over all rows/columns in the matrix
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        for (k = 0; k < n; k++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }

    // stop clock
    clock_gettime(CLOCK_MONOTONIC, &end);

    // calculate time taken for this size
    diff = BILLION * (end.tv_sec - start.tv_sec)
                      + end.tv_nsec - start.tv_nsec;

    // record absolute time or
    // scale by the number of operation which is loop^3, otherwise set to 1
    
    // write to file
    int index = log2(n) - log2(MIN_SIZE);
    arr[index] = diff/(n*n*n);

    // for sanity check print 8 by 8 upper left submatrix of the product c
    // remove for the final code
    // printf("size = %4d diff = %lf \n",n, time);
    // for(i = 0; i < 8; i++) {
    //   for(j = 0; j < 8; j++) {
    //     printf("%8.2e  ",c[i][j]);
    //   }
    //   printf("\n");
    // }
  }

  // writing results from array into .csv file
  for (i = 0; i < 5; i++) {
    fprintf(fp, "\n%d,%f", i+6, arr[i]);
  }

  fclose(fp); free(a); free(b); free(c);

/*
*  Create one way pipe line with call to popen()
*  need tile.csv file and plot_tile.gp file
*/

  FILE *tp = NULL;
  if (( tp = popen("gnuplot plot_rbyc.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }

// Close the pipe
  pclose(tp);

}
