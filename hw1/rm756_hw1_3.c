/*
 * Blocked matrix multiplication, c = a*b,
 *    subblocks of a, b and c are cached.
 * There are two outer loops, one over size of matrices, the other
 * over size of blocks (tiles).
 * Outer loop increases matrix dimension n from MIN_SIZE doubling
 * on each pass until MAX_SIZE is reached.
 * Inner loop increases block size b_s from MIN_BLOCK doubling
 * on each pass untill MAX_BLOCK is reached.
 * It is assumed that n is divisible by b_s.
 * NOTE: for MAX_SIZE = 2^13 it takes many minutes to finish
 *
 * Timing results are stored in file the tile.csv as a 2D array where
 * rows correspond to growing sizes of blocks, and columns correspond
 * to growing dimensions of the matrices plus one. Make the first column
 * the column of indices against which other columns are plotted.
 *
 * compile: gcc -std=gnu99 -O3 -o rm756_hw1_3 rm756_hw1_3.c -lm
 * run code:   ./rm756_hw1_3
 *
 *
 * Additions:
 * (a) add the case when matrix dimension is not divisible by block size
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

// for benchmarking various sizes of matrices and blocks
// set sizes here, otherwise read from the command line
# define MIN_SIZE    1<<6
# define MAX_SIZE    1<<11
# define MIN_BLOCK   1<<3
# define MAX_BLOCK   1<<5
# define BILLION 1000000000L

int main(int argc, char *argv[])
{

   // loop and other indices 
   int i = 0, j = 0, k = 0, b_s = 0, ib = 0, jb = 0, kb = 0;
      
   // variables for timings
   struct timespec start, end;
   // float sec, nsec, exec_time;
   int dim_n = MAX_SIZE;
   int n = MAX_SIZE;

   // open file for recording time measurements
   FILE *fp = NULL;
   fp = fopen("tile.csv", "w");
   
   // define subblocks and allocate memory 
   // float *tile_a, *tile_b, *tile_c;

   // define matrices
   float **a, **b, **c;

   // allocate space for c, use malloc or  calloc
   c = (float **) malloc(n * sizeof(float *));
   
   if(!c) {
      printf("Memory not allocated, malloc returns NULL!\n");
      exit(1);
   }

   for(i = 0; i < n; i++) {
      c[i] = (float *) malloc(n * sizeof(float));
      for(j = 0; j < n; j++) {
            c[i][j] = 0;
      }
   }

   // a and b are set in such a way that it easy to check
   // for corretness of the product see
   // once verified that the code is correct, initialize as needed

   // initialize a
   a = (float **) malloc(dim_n * sizeof(float *));
   for(i = 0; i < dim_n; i++) {
      a[i] = (float *) malloc(dim_n * sizeof(float));
      for(j = 0; j < dim_n; j++) {
            a[i][j] = i*1.0;
      }
   }

   // initialize b
   b = (float **) malloc(dim_n * sizeof(float *));
   for(i = 0; i < dim_n; i++) {
      b[i] = (float *) malloc(dim_n * sizeof(float));
      for(j = 0; j < dim_n; j++) {
            b[i][j] = 1.0*j;
      }
   }

   // output array that will be written to .csv file
   int cols = log2(MAX_BLOCK) - log2(MIN_BLOCK) + 1;
   int rows = log2(MAX_SIZE) - log2(MIN_SIZE);
   float arr[rows][cols];

   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         arr[i][j] = 0;
      }
   }

   // ------- loop from MIN_SIZE, doubling the size, up to MAX_SIZE ----
   for (n = MIN_SIZE; n <= MAX_SIZE; n = n * 2 ) {

      // ------- loop from MIN_BLOCK, doubling the size, up to MAX_BLOCK ----
      for (b_s = MIN_BLOCK; b_s <= MAX_BLOCK; b_s = b_s * 2) {

         // number of blocks in a row/column is the total size divided by block size
         // int block_p = n / b_s;

         // load subblocks of a,b,c to cache, and perform in cache multiplication
         // accumulate products

         // start the clock
         clock_gettime(CLOCK_MONOTONIC, &start); 	

         for (ib=0; ib<n;ib=b_s+ib){
            for (jb=0; jb<n; jb = b_s+jb){
               // load subblock c(ib,jb) into cache as tile_c, done once per each subblock of c
               float tile_c[b_s][b_s];

               // subblocks of a and b are loaded number of subblocks times
               for (kb=0; kb<n; kb = b_s+kb){
                  // kb tells us which block of A/B we are looking at

                  // // load subblock a(ib, kb) into cache as tile_a
                  // float tile_a[b_s][b_s];
                  // for (i = 0; i < b_s; i++) {
                  //    for (k = 0; k < b_s; k++) {
                  //       tile_a[i][k] = a[ib+i][kb+k];
                  //    }
                  // }
                  // // load subblock b(kb,jb) into cache as tile_b
                  // float tile_b[b_s][b_s];
                  // for (k = 0; k < b_s; k++) {
                  //    for (j = 0; j < b_s; j++) {
                  //       tile_b[k][j] = b[kb+k][jb+j];
                  //    }
                  // }

                  // find product tile_c(i,j)
                  for (i = 0; i < b_s; i++) {
                     for (j = 0; j < b_s; j++) {
                        for (k = 0; k < b_s; k++) {
                           tile_c[i][j] += a[i+ib][k+kb] * b[k+kb][j+jb];
                        }
                     }
                  }
               }

               // store tile_c(i,j) back to main memory
               for (i = 0; i < b_s; i++) {
                  for (j = 0; j < b_s; j++) {
                     c[ib+i][jb+j] = tile_c[i][j];
                  }
               }
               
               // stop the clock and measure the multiplication time
               clock_gettime(CLOCK_MONOTONIC, &end);
               float diff = BILLION * (end.tv_sec - start.tv_sec)
                     + end.tv_nsec - start.tv_nsec;

               // write the measurement to file "tile.csv"
               // [matrix size, block size]
               int lm = log2(n) - log2(MIN_SIZE);
               int lb = log2(b_s) - log2(MIN_BLOCK);
               arr[lm][lb] = diff;

               // sanity check, remove from the final code
               // for (i=0; i<8; i++){
               //    for (j=0; j<8; j++){
               //       printf("%8.2e  ",c[i][j]);
               //    }
               //    printf("\n");
               // }    
            }
         }
      } // end of block size loop
   } // end of matrix size loop

   // writing results to .csv file
   int begin = 6;
   for(i=0;i<rows;i++){
      fprintf(fp,"\n%d", begin++);
      for(j=0;j<cols;j++) {
         fprintf(fp,",%f ",arr[i][j]);
      }
   }

   // close the file and free memory
   fclose(fp); free(a); free(b); free(c);

   /*
   *  Create one way pipe line with call to popen()
   *  need tile.csv file and plot_tile.gp file
   */

   FILE *tp = NULL;
   if (( tp = popen("gnuplot plot_tile.gp", "w")) == NULL)
   {
      perror("popen");
      exit(1);
   }
   // Close the pipe
   pclose(tp);
   
   return 0;
}
