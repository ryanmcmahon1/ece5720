/***********************************************************************************  
* Gaussian elimination with partial pivoting on an N by N matrix
*   (1) uses cyclic by row distribution for the elimination steps
*   (2) only a single thread finds a pivot row and swaps it
*       if necessary
*   (3) backsubstitution perfomed by 
*       (a) the main thread if there is only "few" rhs vectors
*           you can set a single rhs by demanding that the solution x
*           is all 1s, then b = A*x
*       (b) collectively by all threads if there are "many" rhs vectors
*           you can set N rhs vectors which are columns of an identity
*           matrix (this will be the case (c) below)
*       (c) finding the inverse will require N rhs vectors
*
*   Possible improvements/changes (not required for this assignment)
*   (1) threads cooperate to find the pivot - requires a mutex which is
*       costly, most likely not worth doing
*   (2) threads cooperate in a single  backsubstitution - must use tiling, 
*       barriers and locks
*       a single thread most likely is faster 
*   (3) after the elimination part is done, one could set strictly lower 
*       traingular part to zero, however to check for correctness you
*       may want to retain numerical values 
*       to see whether they are indeed close to zeroes  
*   (4) consider the case when matrix dimension is not divisible by
*       number of threads
*
*   Benchmarking is done for a range of matrix dimesions and different 
*   number of threads.
*     (a) The outer loop increases matrix dimension N from MIN_DIM, doubling
*         on each pass until MAX_DIM is reached
*     (b) The inner loop increases the number of threads from MIN_THRS to 
*         MAX_THRS doubling on each pass 
*   It is assumed that N is divisible by num_thrs. Feel free to add the case 
*   when N is not divisible by num_thrs.
*
*   NOTE: for MAX_DIM = 2^13 it takes many minutes to finish.
*
*   compile: gcc -std=gnu99 -O3 -o rm756_hw2_code rm756_hw2_code.c -lpthread -lm
*   run: ./rm756_hw2_code
*   NOTE: for matrix inversion, set MIN_DIM = MAX_DIM = Nrhs
*         for solving linear system of equations, set Nrhs = 1
*
*   alternatively use input from the command line and run as
*        ./my_pt_gauss #1 #2 where
*        #1 is the (square) matrix dimension
*        #2 is the number is threads 
************************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <pthread.h>
#include "pthreadbarrier.h"

// set dimensions for testing
# define MIN_DIM     1<<8        // min dimension of the matrices (equal to MIN_DIM if Nrhs=1)
# define MAX_DIM     1<<12        // max dimension (equal to MAX_DIM if Nrhs=1)
# define MIN_THRS    1           // min size of a tile
# define MAX_THRS    8           // max size of a tile
# define Nrhs        1
         // number of rhs vectors (set to MIN_DIM)
# define BILLION 1000000000L

// can be used when N is not divisible by num_thrs
#define min(a, b) (((a) < (b)) ? (a) : (b))

void data_A_b(int N, float** A, float** b);            // create data
void *triangularize(void *arg);                        // triangularization
// void triangularize();
void *backSolve(void *arg);                            // backsubstitution
// void backSolve();
float error_check(float** A, float** x, float** b, int N, int nrhs);       // check residual ||A*x-b||_2
void print_arr(float** arr, int rows, int cols);

pthread_barrier_t barrier;   // used to synchronize threads

// create a global structure visible to all threads,
// the stucture carries all necessary info
struct Thread_Data {
        float** A;           // pointer to matrix A
        float** b;           // pointer to rhs vectors b
        float** x;           // pointer to solution vectors
        int N;               // dimension of A
        int nrhs;            // number of rhs vectors
        int thrs_used;       // number of threds
} thread_data;

/************************* main ********************************/

int main(int argc, char *argv[]) {
 
  /******** loop indices, other helper variables ************/
  int q, ii;

  /********* file writing declarations **********************/
  // would like to benchmark for a range of sizes and different
  // number of threads, and record timings to a file

  FILE *fp = NULL;
  fp = fopen("Gauss_solver.csv", "w");

  /********* timing related declarations **********************/
  struct timeval start, end;     // start and stop timer
  float el_time;                 // elapsed time


  // ---- loop over matrix dimensions N, doubling the sizes at each pass ---
  for (int N = MIN_DIM; N <= MAX_DIM; N = N * 2) {

    // ---- loop over num_thrs, doubling the sizes at each pass ----
    for (int num_thrs = MIN_THRS; num_thrs <= MAX_THRS; num_thrs = num_thrs * 2) {

      /********* thread related declarations **********************/
      // redefined after each pass in the num_thrs loop
      pthread_t thread[num_thrs];
      pthread_barrier_init(&barrier, NULL, num_thrs);
      void *status;

      int ncols = log2(MAX_DIM) - log2(MIN_DIM) + 1;
      int nrows = log2(MAX_THRS) - log2(MIN_THRS);
      float arr[nrows][ncols];
  
      for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
          arr[i][j] = 0;
        }
      }

      // Allocate memory for A
      float **A = (float **)malloc(N*sizeof(float*));
      for (q=0; q < N; q++)
        A[q] = (float*)malloc(N*sizeof(float));

      // Allocate memory for b and x, 
      // for Nrhs = 1 a single rhs, for Nrhs = N for inversion
      float** b = (float**) malloc(sizeof(float*)*N);
      for (q=0; q < N; q++)
        b[q] = (float*)malloc(N*sizeof(float));

      float** x = (float**) malloc(sizeof(float*)*N);
      for (q=0; q < N; q++)
        x[q] = (float*)malloc(N*sizeof(float));

      // set members in thread_data to pass to threads 
      // like thread_data.A = A, etc.
      thread_data.A = A;
      thread_data.b = b;
      thread_data.x = x;
      thread_data.N = N;
      thread_data.nrhs = Nrhs;
      thread_data.thrs_used = num_thrs;

      // used to pass the thread ids to the pthread function, 
      int *index = malloc (num_thrs*sizeof (uintptr_t));
      for(int ii = 0; ii < num_thrs; ii++) {
        index[ii] = ii;
      }

      // populate A and b so the solution x is all 1s
      data_A_b(N,A,b);

      // printf("\nmatrix dimension: %d, number of threads: %d\n", N, num_thrs);

      // start timer
      gettimeofday(&start, NULL);

      // activate threads for triangularization of A and update of b
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_create(&thread[ii], NULL, triangularize, (void *) &index[ii]);
      }

      // terminate threads (join)
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_join(thread[ii], &status);
      }

      // stop timer
      gettimeofday(&end, NULL);

      // get triangularization execution time
      float diff = BILLION * (end.tv_sec - start.tv_sec)
                    + 1000 * (end.tv_usec - start.tv_usec);
      float difft_s = diff / BILLION;

      // backsubstitution, A is now upper triangular, b has changed too
      gettimeofday(&start, NULL);

      // activate threads for backsubstitution 
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_create(&thread[ii], NULL, backSolve, (void *) &index[ii]);
      }

      // terminate threads
      for (ii = 0; ii < num_thrs; ii++) {
        pthread_join(thread[ii], &status);
      }

      // stop timer
      gettimeofday(&end, NULL);

      // get the total execution time
      diff = BILLION * (end.tv_sec - start.tv_sec)
                    + 1000 * (end.tv_usec - start.tv_usec);
      float diffb_s = diff / BILLION;
      el_time = difft_s + diffb_s; // total elapsed time is triangularization time
                                   // plus backsubstitution time
      int ln = log2(N) - log2(MIN_DIM);
      int lt = log2(num_thrs) - log2(num_thrs);
      arr[lt][ln] = el_time;
    

      // check the residual error 
      float err = error_check(A, x, b, N, Nrhs);
      // printf("error: %8.2e\n", err); // commented out for submission

      // free(A); free(b); free(x);// write data from 2d array into .csv file

    } // end of num_thrs loop <-------------------


  } // end of N loop <--------------------

  fclose(fp);

  // NOTE: I used other program to plot results

  return 0;
}

void data_A_b(int N, float** A, float** b){
  int i, j, k;

  // for numerical stability create A as follows
  for (i=0; i<N; i++){
    for (j=0; j<N; j++)
      A[i][j] = 1.0/(1.0*i + 1.0*j + 1.0);

    A[i][i] = A[i][i] + 1.0;
  }

  /* create b, either as columns of the identity matrix, or */
  /* when Nrhs = 1, assume x all 1s and set b = A*x         */
  if (Nrhs == 1) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        for (k = 0; k < N; k++)
          b[i][j] += A[i][k];
      }
    }
  }

  else {
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        if (i == j)
          b[i][j] = 1;
        else
          b[i][j] = 0;
      }
    }
  }
}

void print_arr(float** arr, int rows, int cols) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%.3f ", arr[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void* triangularize(void *arg) {
  int myid = *((int*)arg);
  int i, j, k, m, piv_index, thrs_used;
  /* other variables */
                             
  // copy from global thread_data to local data
  float** A = thread_data.A;
  float** b = thread_data.b;
  thrs_used = thread_data.thrs_used;
  int N = thread_data.N;

  // thread myid finds index piv_indx of pivot row in column i
  // and next swaps rows i and  piv_indx 

  for(i = 0; i<N; i++) {
    if ((i%thrs_used) == (int) myid) {
    // always have thread 0 take care of finding pivot and swapping
    // if (myid == 0) {
      /* your code for finding pivot */
      /* your code for swapping rows i and piv_indx in A and b */
      // printf("\nthread %d finding pivot and swapping rows\n", myid);
      float max = fabsf(A[i][i]);
      piv_index = i;
      for (m = i + 1; m < N; m++) {
        if (fabsf(A[m][i]) > max) {
          max = fabsf(A[m][i]);
          piv_index = m;
        }
      }

      if (piv_index != i) {
        float* temp_A =  A[i];
        A[i] = A[piv_index];
        A[piv_index] = temp_A;

        float* temp_b =  b[i];
        b[i] = b[piv_index];
        b[piv_index] = temp_b;
      }
    }

    // all threads wait until swapping of row i and piv_indx are done
    // print_arr(A, 8, 8);

    pthread_barrier_wait(&barrier);

    // rows i+1 to N can be updated independently by threads 
    // based on cyclic distribution of rows among threads

    for (j = i + 1; j < N; j++) {
      // only do the below work if this thread is assigned to this row
      // each thread does all the rows where j % thrs_used == myid
      if (j % thrs_used == myid) {
        // printf("\nthread %d updating row %d", myid, j);
        float p = A[j][i] / A[i][i];
        for (k = i; k < N; k++) {
          A[j][k] = A[j][k] - (p * A[i][k]);
        }

        for (k = 0; k < thread_data.nrhs; k++)
          b[j][k] = b[j][k] - (p * b[i][k]);
      }
    }

    // wait for all
    pthread_barrier_wait(&barrier);
  }
  pthread_barrier_wait(&barrier);
  
  return 0;
}

void *backSolve(void *arg){
// void backSolve() {
  int myid = *((int*)arg);
  int k = 0;

  // copy global thread_data to local data
  float** A = thread_data.A;
  float** b = thread_data.b;
  float** x = thread_data.x;
  int thrs_used = thread_data.thrs_used;
  int N = thread_data.N;

  // thread myid performs backsubstitution for Nrhs/thrs_used rhs
  // column cyclic distribution
   
  // if Nrhs > 1, split up the work evenly among threads
  // if Nrhs = 1, have thread 0 do the work 
  for(k= myid;k < Nrhs; k += thrs_used){  // loop over # rhs
    // printf("using thread %d", myid);
    for (int i = N - 1; i >= 0; i--) {
      x[i][k] = b[i][k];
      for (int j = i + 1; j < N; j++) {
        x[i][k] = x[i][k] - (x[j][k] * A[i][j]);
      }
      x[i][k] = x[i][k] / A[i][i];
    }
  }
}

float error_check(float** A, float** x, float** b, int N, int nrhs){
  int i, j, k, q;

  /************************************************************************ 
   * compute residual r = b - A*x, compute ||r||_2 = sqrt(sum_i(r[i]*r[i]))
   * compute ||x||_2 = sqrt(sum_i(x[i]*x[i])), 
   * ||A||_F = sqrt(sum_i(sum_j(a[i][j]*a[i][j])))
   * compute normalized residual error res_error 
   *   res_error =  ||r||_2/(||A||_F*||x||_2)
   * in single precision it should be close to 1.0e-6
   * in double precision it should be close to 1.0e-15
   *************************************************************************/
  // multiply A and x
  float **r = (float **)malloc(N*sizeof(float*));
  for (q=0; q < N; q++)
    r[q] = (float*)malloc(Nrhs*sizeof(float));
  // printf("\n eep eep \n");

  for (i = 0; i < N; i++) {
    for (j = 0; j < Nrhs; j++) {
      for (k = 0; k < N; k++) {
        // printf("\n%d, %d, %d\n", i, j, k);
        r[i][j] += A[i][k] * x[k][j];
      }
    }
  }
  
  // print_arr(r, 8, nrhs);
  
  // calculating norm of residual vector r
  float r2_norm = 0;
  for (i = 0; i < N; i++) {
    // r = Ax - b
    r[i][0] = r[i][0] - b[i][0];
    r2_norm += r[i][0] * r[i][0];
  }
  r2_norm = sqrtf(r2_norm);

  // norm of A
  float A_norm = 0;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A_norm += A[i][j] * A[i][j];
    }
  }

  // norm of b
  float x_norm = 0;
  for (i = 0; i < N; i++) {
    for (j = 0; j < nrhs; j++) {
      x_norm += x[i][j] * x[i][j];
    }
  }
  x_norm = sqrtf(x_norm);

  return r2_norm / (A_norm * x_norm);
}

