// HW1 part 1: Ryan McMahon (rm756)

// For this part of the homework I referenced the following website:
// https://www.cs.rutgers.edu/~pxk/416/notes/c-tutorials/gettime.html
// I also referenced this website to help convert a 2D array into .csv:
// http://codingstreet.com/create-csv-file-in-c/

/* 
cache access time by different strides
(1) Set up a linear array of MAX_SIZE floats. 
(2) In a loop, examine access to subarrays of sizes from MIN_SIZE, 
    doubling the size in each pass through the loop, till size MAX_SIZE
(3) For each size of subarrays access their elements with varying length
    strides from stride 1, doubling the stride, until half of the size
    of the current subarray.
(4) Repeate (3) so the current subarray was accessed K*current_size times
    where K is used to average K runs

(5) Store time measurements in an SxL array where
      S is the number of different strides probed plus one
      L is the number of different subarrays used
      (for plotting results, make the first column logarithms base 2 
       of strides)

(6) use gnuplot to plot timing results, "gnuplot cahce_plots.gp"
    where the script cache_plots.gp is provided

    compile: gcc -std=gnu99 -o rm756_hw1_1 rm756_hw1_1.c -lm
    run: ./rm756_hw1_cache                          
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K 		10	        // run each stride K times
#define BILLION 	1000000000
#define min_exp         16              // smallest array tested
#define max_exp         26              // use as large max_exp as 
                                        // the system will tolerate

int main(int argc, char **argv) {

  // define MIN_SIZE and MAZ_SIZE
  int MIN_SIZE = 1 << 10;
  int MAX_SIZE = 1 << 26;

  // for starting and stopping timers
  struct timespec start, end; 

  // dynamic memory allocation for A
  float *A;
  A = (float *) malloc(MAX_SIZE*sizeof(float));
  if(!A) {
    printf("Memory not allocated, malloc returns NULL!\n");
    exit(1);
  }

  float p;     // used to touch A[i] for each loop iteration
  float diff;  // used to calculate elapsed time (end-start)
  int j, i = 0; // initialize loop variables

  // open file for writing timing results
  FILE *tp = NULL;                    
  tp = fopen("time_cache.csv", "w"); // file where time measurements
                                     // are stored

  // get clock resolution
  clock_getres(CLOCK_MONOTONIC, &start);
  printf("resolution of CLOCK_MONOTONIC is %ld ns\n", start.tv_nsec);

  // setting the number of rows and columns in 2d array based on number of strides
  int cols = log2(MAX_SIZE) - log2(MIN_SIZE) + 1;
  int rows = log2(MAX_SIZE);
  float arr[rows][cols];
  
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      arr[i][j] = 0;
    }
  }
	
  // Main loop,  double the array length starting from min length

  for (int n = MIN_SIZE; n <= MAX_SIZE; n = 2 * n) {

    // Loop over strides (doubling until it is half of n)
    for (int s = 1; s <= n / 2; s = 2 * s) {

      // start timer
      clock_gettime(CLOCK_MONOTONIC, &start);

      // repeat K times the current stride s for averaging
      for (j = 0; j < K*s; j++) { 
        for (i = 0; i < n; i+=s)
          /* access A[i] */
          p = A[i];
      }
      // stop timer
      clock_gettime(CLOCK_MONOTONIC, &end);

      // compute the average access time in K repetitions
      diff = BILLION * (end.tv_sec - start.tv_sec)
                      + end.tv_nsec - start.tv_nsec;

      // write to 2d rray based on the current array size and stride
      // (going down column gives higher strides, going down row is higher array size)
      float avg = diff / (n * K);
      int ln = log2(n) - log2(MIN_SIZE);
      int ls = log2(s);
      arr[ls][ln] = avg;

    // end of loop over strides
    }
  // end of loop over subarrays
  }

  // int begin = log2(MIN_SIZE);
  int begin = 1;

  // write data from 2d array into .csv file
  for(i=0;i<rows;i++){
 
    fprintf(tp,"\n%d", begin++);
 
    for(j=0;j<cols;j++) {
        fprintf(tp,",%f ",arr[i][j]);
    }
  }

// close file and free A
  fclose(tp); free(A);

/* -------- this part is to create a gnuplot *.eps graph ----------- */
/* the name is set to cache_plots.eps but you are free to change it  */                           

  FILE *fp = NULL;            // script for gnuplot which generates
                              // eps graph of time measurements

/* Create one way pipe line with call to popen() */
  if (( fp = popen("gnuplot plot_cache.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }

/* Close the pipe */
  pclose(fp); 

  return(0);

}
