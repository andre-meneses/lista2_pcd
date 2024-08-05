#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void Read_input(int rank, int* vetor1, int* vetor2, int *scalar, int array_size, int local_array_size, int* local_vetor1, int* local_vetor2, MPI_Comm comm);
void Parallel_mult_dot(int* local_vetor1, int* local_vetor2, int* local_result, int local_array_size, int scalar, MPI_Comm comm);

int main() {
    int array_size = 6; // Example where array_size is not a multiple of comm_sz
    int* vetor1 = malloc(array_size * sizeof(int));
    int* vetor2 = malloc(array_size * sizeof(int));
    int scalar;
    int local_result;

    int comm_sz;
    int rank;

    MPI_Init(NULL, NULL); 

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    int* sendcounts = malloc(comm_sz * sizeof(int));
    int* displs = malloc(comm_sz * sizeof(int));

    int base_size = array_size / comm_sz;
    int remainder = array_size % comm_sz;

    for (int i = 0; i < comm_sz; i++) {
        sendcounts[i] = base_size;
        if (i < remainder) {
            sendcounts[i]++;
        }
    }

    displs[0] = 0;
    for (int i = 1; i < comm_sz; i++) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }

    int local_array_size = sendcounts[rank];
    int* local_vetor1 = malloc(local_array_size * sizeof(int));
    int* local_vetor2 = malloc(local_array_size * sizeof(int));

    Read_input(rank, vetor1, vetor2, &scalar, array_size, local_array_size, local_vetor1, local_vetor2, MPI_COMM_WORLD);

    MPI_Scatterv(vetor1, sendcounts, displs, MPI_INT, local_vetor1, local_array_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(vetor2, sendcounts, displs, MPI_INT, local_vetor2, local_array_size, MPI_INT, 0, MPI_COMM_WORLD);

    Parallel_mult_dot(local_vetor1, local_vetor2, &local_result, local_array_size, scalar, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result: %d \n", local_result);
    }

    // Clean up
    free(vetor1);
    free(vetor2);
    free(local_vetor1);
    free(local_vetor2);
    free(sendcounts);
    free(displs);
    MPI_Finalize();

    return 0;
}

void Read_input(int rank, int* vetor1, int* vetor2, int *scalar, int array_size, int local_array_size, int* local_vetor1, int* local_vetor2, MPI_Comm comm) {
    if(rank == 0) {
        printf("Escalar: ");
        scanf("%d", scalar);
        printf("\n");

        printf("Vetor 1:\n");
        for(int i = 0; i < array_size; i++)
            scanf("%d", &vetor1[i]);

        printf("Vetor 2:\n");
        for(int i = 0; i < array_size; i++)
            scanf("%d", &vetor2[i]);
    }

    MPI_Bcast(scalar, 1, MPI_INT, 0, comm);
}

void Parallel_mult_dot(int* local_vetor1, int* local_vetor2, int* local_result, int local_array_size, int scalar, MPI_Comm comm) {
    int local_dot_product = 0;

    for(int i = 0; i < local_array_size; i++) {
        local_dot_product += (scalar * local_vetor1[i]) * local_vetor2[i];
    }

    MPI_Reduce(&local_dot_product, local_result, 1, MPI_INT, MPI_SUM, 0, comm);
}

