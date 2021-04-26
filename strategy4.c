#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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

void write_output_Transpose(char fname[], double** arr, int n ){
	FILE *f = fopen(fname, "w");
	for( int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			fprintf(f, "%0.12f ", arr[j][i]);
		} 
		fprintf(f, "\n");
	}
	fclose(f);
}

void print_matrix(double **A,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[i][j]);
		}
		printf("\n");
	}
}

void print_matrix2(double A[4][4],int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[i][j]);
		}
		printf("\n");
	}
}

void print_matrix_transpose(double **A,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[j][i]);
		}
		printf("\n");
	}
}

int main(int argc,char* argv[]){

	double start_main,end_main,start,end;

	int n,N,num_threads;
	int comm_sz;	// number of process
	int my_rank;	// my process rank


	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	if(my_rank==0){
		start_main = MPI_Wtime();
	}

	N = atoi(argv[1]);
	num_threads = atoi(argv[3]);


	// plan is to add padding so that scatter/gather works
	int padding = N-N%num_threads;
	if(padding==N)padding=0;

	if(comm_sz>N){
		padding = comm_sz-N;
	}

	n = N + padding;

	double (*A)[n] = (double(*)[n])malloc(sizeof(*A)*n);
	double (*L)[n] = (double(*)[n])malloc(sizeof(*L)*n);
	double (*U)[n] = (double(*)[n])malloc(sizeof(*U)*n);



	if(my_rank==0){
		
		char* input = argv[2];
		
		FILE* f = fopen(input,"r");
		for(int i=0;i<N;i++){
			for(int j=0;j<N;j++){
				fscanf(f,"%lf",&A[i][j]);
			}
		}
		for(int i=N;i<n;i++){
			for(int j=N;j<n;j++){
				A[i][j]=0;
			}
		}

		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				L[i][j]=0;
				if(i<N && i==j){
					U[i][j]=1;
				}
				else{
					U[i][j]=0;
				}
			}
		}

		// take transpose of U
	}	

	MPI_Bcast(&(A[0][0]),n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	if(my_rank==0){
		start = MPI_Wtime();
	}
	// CROUT MATRIX DECOMPOSITION STARTS HERE

	int i, j, k;
	double sum = 0;

	int chunk = n/comm_sz;

	double (*sub_L)[n] = (double(*)[n])malloc(sizeof(*sub_L)*chunk);
	double (*sub_U)[n] = (double(*)[n])malloc(sizeof(*sub_U)*chunk);
	int left_i,right_i;
	MPI_Barrier(MPI_COMM_WORLD);

	for (j=0; j<N; j++) {
		// First do with scatter, then try with scatterv() later.
		MPI_Scatter(&(L[0][0]),chunk*n,MPI_DOUBLE,sub_L,chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(U[j],n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		/* Allocating iteration ranges */
		if(my_rank*chunk>=j){
			left_i=0;
			if(my_rank*chunk>=N){
				right_i=-1;
			}
			else if(my_rank*chunk+chunk>=N){
				right_i=N-my_rank*chunk;
			}
			else{
				right_i=chunk;
			}
		}
		else{
			if(j-my_rank*chunk<chunk){
				left_i=j-my_rank*chunk;
				right_i=chunk;
			}
			else{
				left_i=0;
				right_i=-1;
			}
		}


		// L's Loop
		for (i = left_i; i < right_i; i++) {
			sum = 0;
			for (k = 0; k < j; k++) {
				sum = sum + sub_L[i][k] * U[j][k];
			}
			sub_L[i][j]=A[my_rank*chunk+ i][j] - sum;
		}

		MPI_Gather(sub_L,chunk*n,MPI_DOUBLE,&(L[0][0]),chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// Checking for L[j][j]==0 once
		if(my_rank==0){
			if (L[j][j] == 0) {
				printf("-------------------------------------TERMINATED\n");
				MPI_Abort(MPI_COMM_WORLD,0);
			} 
		}
		
		MPI_Scatter(&(U[0][0]),chunk*n,MPI_DOUBLE,sub_U,chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(L[j],n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// U's Loop
		for (i=left_i; i<right_i; i++) {
			sum = 0;
			for(k = 0; k < j; k++) {
				sum = sum + L[j][k] * sub_U[i][k];
			} 
			sub_U[i][j] =(A[j][chunk*my_rank+i] - sum) / L[j][j];
		}
		MPI_Gather(sub_U,chunk*n,MPI_DOUBLE,&(U[0][0]),chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

		// Placed barrier at the end, so that all processes complete before going to next loop
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// CROUT MATRIX DECOMPOSITION ENDS HERE

	// Convert double[][] to double** here before calling print_matrix()
	// scatter only works with double[][]
	// if we try double**, then 1. not contiguous, 2. we might place burden on process 0 to scatter data.

	if(my_rank==0){
		end = MPI_Wtime();
		printf("strategy 4 CROUT: %0.12f\n",end-start);
	}

	if(my_rank==0){

		double* A1[N],*L1[N],*U1[N];
		for(i=0;i<N;i++){
			A1[i] = (double*)malloc(sizeof(double)*N);
			L1[i] = (double*)malloc(sizeof(double)*N);
			U1[i] = (double*)malloc(sizeof(double)*N);
		}

		for(i=0;i<N;i++){
			for(j=0;j<N;j++){
				A1[i][j]=A[i][j];
				L1[i][j]=L[i][j];
				U1[i][j]=U[j][i];
			}
		}


		char str[100] = "output_L_4_";
		char intbuf[10];
		sprintf(intbuf,"%d",comm_sz);
		strcat(str,intbuf);
		strcat(str,".txt");
		write_output(str,L1,N);
		char str2[100] = "output_U_4_";
		strcat(str2,intbuf);
		strcat(str2,".txt");
		write_output(str2,U1,N);

	}
	  
	if(my_rank==0){
		end_main = MPI_Wtime();
		printf("strategy 4 MAIN: %0.12f\n",end_main-start_main);
	}

	MPI_Finalize();
	return 0;
}