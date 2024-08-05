#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

void process_zero(MPI_Datatype upper_triangular, int n);
void process_one(MPI_Datatype upper_triangular, int count);

int main() {
    int comm_sz, rank;
    MPI_Init(NULL, NULL); 
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    int n;
    int count;

    if (rank == 0) {
        printf("Order of the matrix: ");
        scanf("%d", &n);
        count = n * (n + 1) / 2;  // Calculate the number of elements in the upper triangular part
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int *blocklengths = (int *) malloc(count * sizeof(int));
    int *displacements = (int *) malloc(count * sizeof(int));

    int idx = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            blocklengths[idx] = 1;
            displacements[idx] = i * n + j;
            idx++;
        }
    }

    MPI_Datatype upper_triangular;
    MPI_Type_indexed(count, blocklengths, displacements, MPI_INT, &upper_triangular);
    MPI_Type_commit(&upper_triangular);

    if (rank == 0) {
        process_zero(upper_triangular, n);
    } else if (rank == 1) {
        process_one(upper_triangular, count);
    }

    MPI_Type_free(&upper_triangular);
    free(blocklengths);
    free(displacements);

    MPI_Finalize();
    return 0;
}

void process_zero(MPI_Datatype upper_triangular, int n) {
    int* matrix = (int *) malloc(sizeof(int) * (n) * (n));

    printf("Elements of the matrix:\n");
    for (int i = 0; i < (n) * (n); i++) {
        scanf("%d", &matrix[i]);
    }
    
    MPI_Send(matrix, 1, upper_triangular, 1, 0, MPI_COMM_WORLD);

    free(matrix);
}

void process_one(MPI_Datatype upper_triangular, int count) {
    int *recv_buffer = (int *) malloc(count * sizeof(int));

    MPI_Recv(recv_buffer, count, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Upper triangular part: ");
    for (int i = 0; i < count; i++) {
        printf("%d ", recv_buffer[i]);
    }
    printf("\n");

    free(recv_buffer);
}

