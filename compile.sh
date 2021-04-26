gcc -std=c99 -O0 -o 0 -fopenmp strategy0.c
gcc -std=c99 -O0 -o 1 -fopenmp strategy1.c
gcc -std=c99 -O0 -o 2 -fopenmp strategy2.c
gcc -std=c99 -O0 -o 3 -fopenmp strategy3.c
mpicc -o 4 strategy4.c