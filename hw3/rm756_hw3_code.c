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
// run: ./rm756_hw3_code

#define SOFT 1e-2f
#define NUM_BODY 5 /* the number of bodies                  */
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
  
  int i, j;
  int nBodies = NUM_BODY;
  // nBodies = atoi(argv[1]);

  const double dt = 0.01f; // time step
  const int nIters = 1000;  // simulation iterations
  double totalTime, avgTime;

  double *Bbuf = (double*)malloc(nBodies*sizeof(Body));
  double *Ebuf = (double*)malloc(sizeof(Energy));
  Body *r = (Body*) Bbuf;
  Energy *e = (Energy*)Ebuf;

  /******************** (0) initialize N-body ******************/
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
  for (i = 0; i < nBodies; i++) {
    printf("Particle %d: Mass = %f, Position = (%f,%f), Velocity = (%f,%f)\n", i, r[i].m, r[i].x, r[i].y, r[i].vx, r[i].vy);
  }
  /******************** (2) Get total energy *********************/
  total_energy(r, e, nBodies);
  // printf("%f\n", e->ke);

  /******************** (3) Initial acceleration *****************/
  bodyAcc(r, dt, nBodies);            

  totalTime = omp_get_wtime();

  /******************** Main loop over iterations **************/
  for (int iter = 1; iter <= nIters; iter++) {

    // "half kick"
    for (i = 0; i < nBodies; i++) {
      r[i].vx = r[i].vx + (r[i].ax * dt / 2);
      r[i].vy = r[i].vy + (r[i].ay * dt / 2);
    }
        
    // "drift"
    for (i = 0; i < nBodies; i++) {
      r[i].x = r[i].x + (r[i].vx * dt / 2);
      r[i].y = r[i].y + (r[i].vy * dt / 2);
    }

    // update acceleration
    // compute new force matrix

    // "half kick"
    for (i = 0; i < nBodies; i++) {
      r[i].vx = r[i].vx + (r[i].ax * dt / 2);
      r[i].vy = r[i].vy + (r[i].ay * dt / 2);
    }

    // record the position and velocity

    // sanity check
    total_energy(r, e, nBodies);
    if (iter%10 == 0){
      fprintf(tp,"%4d %10.3e %10.3e %10.3e\n",
             iter,(*e).pe,(*e).ke, (*e).pe-(*e).ke);
    }
  }
  
  totalTime = omp_get_wtime() - totalTime;

  free(Bbuf); free(Ebuf);
  fclose(tp); fclose(pvp);
  
  FILE *fp = NULL;          // script for gnuplot which generates
                              // eps graph of timings
  /* Create one way pipe line with call to popen() */
  if (( fp = popen("gnuplot plot_nbody.gp", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }
  /* Close the pipe */
  pclose(fp); 

  /* on Mac only */
  FILE *fpo = NULL;
  if (( fpo = popen("open plot_nbody.eps", "w")) == NULL)
  {
    perror("popen");
    exit(1);
  }
  pclose(fpo);

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
  // potential energy : (*e).pe = -\sum_{1\leq i < j \leq N}G*m_i*m_j/||r_j-r_i||

  double ke = 0.0, pe = 0.0;

  for (int i = 0; i < n; i++) {
    double v = sqrtf((r[i].vx * r[i].vx) + (r[i].vy * r[i].vy));
    ke += r[i].m * v * v;
  }

  e->ke = ke / 2;
  e->pe = 0;
}

void bodyAcc(Body *r, double dt, int n) {
  // F = G*sum_{i,j} m_i*m_j*(r_j-r_i)/||r_j-r_i||^3
  // F = m*a thus a = F/m // constructing initial force matrices (need 2 for 2 dimensions)
  double f_x[n][n]; // forces in x direction
  double f_y[n][n]; // forces in y direction

  // initializing all values to 0
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      f_x[i][j] = 0.0;
      f_y[i][j] = 0.0;
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      double dx = r[i].x - r[j].x;
      double dy = r[i].y - r[j].y;
      // printf("dxdx + dydy = %f,%f ... %f\n", dx*dx, dy*dy, SOFT);
      double d = sqrtf((dx * dx) + (dy * dy) + SOFT); // SOFT added to avoid divide by 0
      double d3 = d * d * d;
      f_x[i][j] = -G * r[i].m * r[j].m * (r[i].x - r[j].x) / d3;
      f_y[i][j] = -G * r[i].m * r[j].m * (r[i].y - r[j].y) / d3;
      f_x[j][i] = -f_x[i][j];
      f_y[j][i] = -f_y[i][j]; // using symmetry of force matrix (Newton's law)
    }
  }
  printf("Forces in x direction:\n");
  print_arr(f_x, n, n);
  printf("Forces in y direction:\n");
  print_arr(f_y, n, n);  

  // calculate initial acceleration from force matrices
  for (int i = 0; i < n; i++) {
    double Fx = 0.0, Fy = 0.0;
    for (int j = 0; j < n; j++) {
      Fx += f_x[i][j];
      Fy += f_y[i][j];
    }
    // printf("Force for particle %d: (%f, %f)\n", i, Fx, Fy);
    // Using F = m * a, a = F / m
    r[i].ax = Fx / r[i].m;
    r[i].ay = Fy / r[i].m;
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
