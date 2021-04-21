#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // TODO: Check: is the include allowed?

void write_output(char fname[], double** arr, int n ){
    FILE *f = fopen(fname, "w");
    for( int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            fprintf(f, "%0.12f ", arr[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void crout(double const **A, double **L, double **U, int n) {
    // Let us write the code for only 2 threads first
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    int chunk1;
    int chunk2;

    for (j = 0; j < n; j++) {
        // Compute L[j][j] first!
        sum = 0;
        for (k = 0; k < j; k++) {
            sum = sum + L[j][k] * U[k][j];
        }
        L[j][j] = A[j][j] - sum;

        if (L[j][j] == 0) {
            exit(0);
        }

        chunk1 = (n-j-1)/8;
        chunk2 = (n-j)/8;

        // plan to divide into 8 sections for each of L's loop and U's loop


    #pragma omp parallel
            {
    #pragma omp sections
            {
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1; i1 < j+1+chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+chunk1; i1 < j+1+2*chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+2*chunk1; i1 < j+1+3*chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+3*chunk1; i1 < j+1+4*chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+4*chunk1; i1 < j+1+5*chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+5*chunk1; i1 < j+1+6*chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+6*chunk1; i1 < j+1+7*chunk1; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
            {
                    int i1;
                    for (i1 = j+1+7*chunk1; i1 < n; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
            }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j; i2 < j+chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+chunk2; i2 < j+2*chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+2*chunk2; i2 < j+3*chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+3*chunk2; i2 < j+4*chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+4*chunk2; i2 < j+5*chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+5*chunk2; i2 < j+6*chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+6*chunk2; i2 < j+7*chunk2; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
        #pragma omp section
                {
                    int i2;
                    for (i2 = j+7*chunk2; i2 < n; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
            }
        }
    }
}

int main(int argc,char* argv[]){

    // gcc -o 0 -fopenmp strategy0.c
    double start_main = omp_get_wtime();

    int n,NUM_THREADS,loop2_threads,loop3_threads;
    n = atoi(argv[1]);
    char* input = argv[2];
    NUM_THREADS = atoi(argv[3]);

    double* A[n];
    double* L[n];
    double* U[n];

    // should we parallelize?
    for(int i=0;i<n;i++){
        A[i] = (double*)malloc(sizeof(double)*n);
        L[i] = (double*)malloc(sizeof(double)*n);
        U[i] = (double*)malloc(sizeof(double)*n);
    }

    FILE* f = fopen(input,"r");

    // now take input
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            fscanf(f,"%lf",&A[i][j]);
        }
    }

    // initialise L,U
    // should we parallelize?

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            L[i][j]=0;
            U[i][j]=0;
        }
    }

    omp_set_num_threads(NUM_THREADS);
//    printf("Now Executing Crout!\n");
    double start,end;
    start = omp_get_wtime();
    crout((const double **) A, L, U, n);

    end = omp_get_wtime();
    printf("strategy 2 CROUT: %0.12f\n",end-start);

    char str[100] = "output_L_2_";
    char intbuf[10];
    sprintf(intbuf,"%d",NUM_THREADS);
    strcat(str,intbuf);
    strcat(str,".txt");

    write_output(str,L,n);

    char str2[100] = "output_U_2_";
    strcat(str2,intbuf);
    strcat(str2,".txt");
    write_output(str2,U,n);

    double end_main = omp_get_wtime();
    printf("strategy 2 MAIN : %0.12f s\n",end_main-start_main);
}