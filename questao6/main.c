#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void generate_random_array(int* array, int count, int seed);
void print_prefix_sums(int* prefix_sums, int count, int rank);

int main() {
    int rank, comm_sz;
    int count = 10;  

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int seed = (1251 * (rank + 2)) % 63;
    int* array = malloc(count * sizeof(int));
    int* prefix_sums = malloc(count * sizeof(int));

    generate_random_array(array, count, seed);

    MPI_Scan(array, prefix_sums, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    print_prefix_sums(prefix_sums, count, rank);

    free(array);
    free(prefix_sums);
    MPI_Finalize();

    return 0;
}

void generate_random_array(int* array, int count, int seed) {
    srand(seed);
    for (int i = 0; i < count; i++) {
        array[i] = rand() % 500;  
    }
}

void print_prefix_sums(int* prefix_sums, int count, int rank) {
    printf("Processo %d prefix sums: ", rank);
    for (int i = 0; i < count; i++) {
        printf("%d ", prefix_sums[i]);
    }
    printf("\n");
}

