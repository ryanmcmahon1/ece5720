#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// references:
// https://www.geeksforgeeks.org/generating-random-number-range-c/
// -for random number generation in a range

// compile: gcc -std=gnu99 -O3 -o rm756_hw3_code rm756_hw3_code.c -fopenmp -lm
// debug: gcc -std=gnu99 -O0 -o rm756_hw3_code rm756_hw3_code.c -fopenmp -lm -g
// run: ./rm756_hw3_code num_bodies num_threads

#define SOFT 1e-2f
#define NUM_BODY 50 /* the number of bodies                  */
#define NUM_THRS 1
#define MAX_X 90    /* the positions (x_i,y_i) are random    */
#define MAX_Y 50 
#define MIN_X 1     /* between MIN and MAX position          */
#define MIN_Y 1  
#define MAX_M 60    /* the masses m_i are random             */
#define MIN_M 1     /* between MIN and MAX mass              */
#define MAX_V 20    /* the velocities (vx_i,vy_i) are random */
#define MIN_V 0     /* between MIN and MAX velocity          */
#define G 0.1       /* gravitational constant                */
#define DIM 2       /* 2 or 3 dimensions                     */
#define MIN_THRS 1
#define MAX_THRS 8
#define MIN_N 1000
#define MAX_N 1001
#define NUM_ITERS 1000

// position (x,y), velocity (vx,vy), acceleration (ax,ay), mass 
typedef struct { double x, y, vx, vy, ax, ay, m; } Body;

// kinetic and potential energy
typedef struct { double ke, pe; } Energy;

void print_arr(double arr[NUM_BODY][NUM_BODY], int rows, int cols);
void bodyAcc(Body *r, double dt, int n);      // computes  a
void total_energy(Body *r, Energy *e, int n); // kinetic and potential
void center_of_momentum(Body *r, int n);      // center of momentum

int main(const int argc, const char** argv) {

  // record positions and velocities
  FILE *pvp = NULL;            
  pvp = fopen("pos_vel.txt", "w");

  // file to record energy (to check for correctness)
  FILE *tp = NULL;            
  tp = fopen("plot_nbody.csv", "w");

  // for (int nBodies = MIN_N; nBodies < MAX_N; nBodies = nBodies * 2) {
  //   for (int p = MIN_THRS; p <= MAX_THRS; p = p*2) {
  //     if (p > 10) {
  //       p = 10; // p should have value 1, 2, 4, 8, 10
  //     }
  int i, j;
  // int nBodies = NUM_BODY;
  int nBodies = atoi(argv[1]);
  int p = atoi(argv[2]);
  // int p = NUM_THREADS;
  // printf("%d, %d\n\n", nBodies, p);

  const double dt = 0.01f; // time step
  const int nIters = 1000;  // simulation iterations
  double totalTime, avgTime;

  double *Bbuf = (double*)malloc(nBodies*sizeof(Body));
  double *Ebuf = (double*)malloc(sizeof(Energy));
  Body *r = (Body*) Bbuf;
  Energy *e = (Energy*)Ebuf;

  /******************** (0) initialize N-body ******************/
  // seeding random number generator so errors are reproducible
  srand(0);
  for (i = 0; i < nBodies; i++) {
    r[i].m  = (rand() % (MAX_M - MIN_M + 1)) + MIN_M;
    r[i].x  = (rand() % (MAX_X - MIN_X + 1)) + MIN_X;
    r[i].y  = (rand() % (MAX_Y - MIN_Y + 1)) + MIN_Y;
    r[i].vx = (rand() % (MAX_V - MIN_V + 1)) + MIN_V;
    r[i].vy = (rand() % (MAX_V - MIN_V + 1)) + MIN_V;
  }

  /******************** (1) Get center of mass *******************/
  center_of_momentum(r, nBodies);
  // for (i = 0; i < nBodies; i++) {
  //   printf("Particle %d: Mass = %f, Position = (%f,%f), Velocity = (%f,%f)\n", i, r[i].m, r[i].x, r[i].y, r[i].vx, r[i].vy);
  // }
  /******************** (2) Get total energy *********************/
  total_energy(r, e, nBodies);

  /******************** (3) Initial acceleration *****************/
  bodyAcc(r, dt, nBodies);

  double F_x[nBodies];
  double F_y[nBodies];

  totalTime = omp_get_wtime();

  /******************** Main loop over iterations **************/
  omp_set_num_threads(p);

  // local copies of force arrays (first index is for thread id)
  double F_xloc[p][nBodies];
  double F_yloc[p][nBodies];
  int t_id;

  for (int iter = 1; iter <= nIters; iter++) {
    #pragma omp parallel shared(F_x, F_y, F_xloc, F_yloc, r) private(i, j, t_id)
    {
      double dx, dy, d, d3; // used for calculating distances between particles
      t_id = omp_get_thread_num();
      int i = 0, j = 0;
      // "half kick"s
      #pragma omp for schedule(auto)
      for (i = 0; i < nBodies; i++) {
        r[i].vx = r[i].vx + (r[i].ax * dt / 2);
        r[i].vy = r[i].vy + (r[i].ay * dt / 2);
      }
          
      // "drift"
      #pragma omp for schedule(auto) 
      for (i = 0; i < nBodies; i++) {
        r[i].x = r[i].x + (r[i].vx * dt);
        r[i].y = r[i].y + (r[i].vy * dt);
      }

      // update acceleration

      // each thread initializes its local force array to 0
      #pragma omp for schedule(auto) 
      for (int i = 0; i < nBodies; i++) {
        F_xloc[t_id][i] = 0.0;
        F_yloc[t_id][i] = 0.0;
      }

      // calculate force on each particle
      #pragma omp for schedule(auto)
      for (int i = 0; i < nBodies; i++) {
        for (int j = i + 1; j < nBodies; j++) {
          dx = r[i].x - r[j].x;
          dy = r[i].y - r[j].y;
          // SOFT added to avoid divide by 0
          d = sqrtf((dx * dx) + (dy * dy) + SOFT);
          d3 = d * d * d;
          double f_xij = -G * r[i].m * r[j].m * (r[i].x - r[j].x) / d3;
          double f_yij = -G * r[i].m * r[j].m * (r[i].y - r[j].y) / d3;
          F_xloc[t_id][i] += f_xij;
          F_yloc[t_id][i] += f_yij;
          // using symmetry from Newton's laws
          F_xloc[t_id][j] -= f_xij;
          F_yloc[t_id][j] -= f_yij;
        }
      }

      // accumulate local forces into total force array
      #pragma omp for schedule(auto) 
      for (int i = 0; i < nBodies; i++) {
        F_x[i] = 0.0;
        F_y[i] = 0.0;
        for (int t = 0; t < p; t++) {
          F_x[i] += F_xloc[t_id][i];
          F_x[i] += F_xloc[t_id][i];
        }
      }

      // calculate acceleration from force matrices
      #pragma omp for schedule(auto) 
      for (int i = 0; i < nBodies; i++) {
        // Using F = m * a, a = F / m
        r[i].ax = F_x[i] / r[i].m;
        r[i].ay = F_y[i] / r[i].m;
      }

      // "half kick"
      #pragma omp for schedule(auto) 
      for (i = 0; i < nBodies; i++) {
        r[i].vx = r[i].vx + (r[i].ax * dt / 2);
        r[i].vy = r[i].vy + (r[i].ay * dt / 2);
      }
    }

    // serial section
    // record the position and velocity
    // printf("\nIteration %d\n", iter);
    // for (i = 0; i < nBodies; i++) {
    //   printf("Position for particle %d: (%f, %f)\n", i, r[i].x, r[i].y);
    // }

    // sanity check
    total_energy(r, e, nBodies);
    // printf("%f + %f = %f\n", e->pe, e->ke, e->pe - e->ke);
    if (iter%100 == 0){
      // printf("%f + %f = %f\n", e->pe, e->ke, e->pe - e->ke);
      fprintf(tp,"%4d %10.3e %10.3e %10.3e\n",
            iter,(*e).pe,(*e).ke, (*e).pe-(*e).ke);
    }
  }
  
  totalTime = omp_get_wtime() - totalTime;
  printf("Total time for %d particles and %d threads = %f\n\n", nBodies, p, totalTime);
  free(Bbuf); free(Ebuf);
  //   }
  // }
  fclose(tp); fclose(pvp);
  
  // FILE *fp = NULL;          // script for gnuplot which generates
  //                             // eps graph of timings
  // /* Create one way pipe line with call to popen() */
  // if (( fp = popen("gnuplot plot_nbody.gp", "w")) == NULL)
  // {
  //   perror("popen");
  //   exit(1);
  // }
  // /* Close the pipe */
  // pclose(fp); 

  // /* on Mac only */
  // FILE *fpo = NULL;
  // if (( fpo = popen("open plot_nbody.eps", "w")) == NULL)
  // {
  //   perror("popen");
  //   exit(1);
  // }
  // pclose(fpo);

  return(0);

}

void center_of_momentum(Body *r, int n){
  /*
  velocity v' in the center-of-momentum reference frame is
  v' = v - V_C where V_C = \sum_i(m_iv_i)/sum_i(m_i)
  (1) the total momentum in the center-of-momentum system venishes
  \sum_i(m_iv'_i) = \sum_i(m_i(v-V_C)) =
  \sum_i(m_iv_i) - \sum_i(m_i(\sum_j(m_jv_j)/(\sum_jm_j)) = 0
  (2) center of the mass C = \sum_i(m_ir_i)/\sum_(m_i) move with
  constant velocity
  https://en.wikipedia.org/wiki/N-body_problem
  https://en.wikipedia.org/wiki/Center-of-momentum_frame
  */

  // calculating total mass and x/y momentum 
  double total_mass = 0.0, x_momentum = 0.0, y_momentum = 0.0;
  for (int i = 0; i < n; i++) {
    total_mass += r[i].m;
    x_momentum += r[i].m * r[i].vx;
    y_momentum += r[i].m * r[i].vy;
  }

  // calculating Vc
  double Vc_x = x_momentum / total_mass;
  double Vc_y = y_momentum / total_mass;

  // printf("%f,%f", Vc_x, Vc_y);

  // applying the transformation v' = v - Vc to each particle
  for (int i = 0; i < n; i++) {
    r[i].vx = r[i].vx - Vc_x;
    r[i].vy = r[i].vy - Vc_y;
  }
}

void total_energy(Body *r, Energy *e, int n){
  // kinetic energy, (*e).ke = m*v^2/2;
  // potential energy : (*e).pe = -\sum_{1\leq i < j \leq N-1}G*m_i*m_j/||r_j-r_i||

  double ke = 0.0, pe = 0.0;
  double dx = 0.0, dy = 0.0;

  for (int i = 0; i < n; i++) {
    double v = sqrtf((r[i].vx * r[i].vx) + (r[i].vy * r[i].vy));
    ke += r[i].m * v * v;
    for (int j = i + 1; j < n; j++) {
      dx = r[i].x - r[j].x;
      dy = r[i].y - r[j].y;
      pe +=  r[i].m * r[j].m / sqrtf((dx * dx) + (dy * dy) + SOFT);
    }
  }

  e->ke = ke / 2;
  e->pe = -G * pe;
}

void bodyAcc(Body *r, double dt, int n) {
  // F = G*sum_{i,j} m_i*m_j*(r_j-r_i)/||r_j-r_i||^3
  // F = m*a thus a = F/m 
  // constructing force matrices (need 2 for 2 dimensions)
  double F_x[n]; // forces in x direction
  double F_y[n]; // forces in y direction

  // initializing all values to 0
  for (int i = 0; i < n; i++) {
    F_x[i] = 0.0;
    F_y[i] = 0.0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dx = r[i].x - r[j].x;
      double dy = r[i].y - r[j].y;
      // printf("dxdx + dydy = %f,%f ... %f\n", dx*dx, dy*dy, SOFT);
      double d = sqrtf((dx * dx) + (dy * dy) + SOFT); // SOFT added to avoid divide by 0
      double d3 = d * d * d;
      double f_xij = -G * r[i].m * r[j].m * (r[i].x - r[j].x) / d3;
      double f_yij = -G * r[i].m * r[j].m * (r[i].y - r[j].y) / d3;
      F_x[i] += f_xij;
      F_y[i] += f_yij;
      F_x[j] -= f_xij;
      F_y[j] -= f_yij;
    }
  }
  // printf("Forces in x direction:\n");
  // print_arr(f_x, n, n);
  // printf("Forces in y direction:\n");
  // print_arr(f_y, n, n);  

  // calculate initial acceleration from force matrices
  for (int i = 0; i < n; i++) {
    // printf("Force for particle %d: (%f, %f)\n", i, Fx, Fy);
    // Using F = m * a, a = F / m
    r[i].ax = F_x[i] / r[i].m;
    r[i].ay = F_y[i] / r[i].m;
  }
}

// print out a 2d array
void print_arr(double arr[NUM_BODY][NUM_BODY], int rows, int cols) {
  int i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      printf("%f ", arr[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}
