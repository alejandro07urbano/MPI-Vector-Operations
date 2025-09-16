#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> 


void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm);
void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);
void Read_data(double local_vec1[], double local_vec2[], double* scalar_p, int local_n, int my_rank, int comm_sz, MPI_Comm comm);
void Print_vector(double local_vec[], int local_n, int n, char title[], int my_rank, MPI_Comm comm);
double Par_dot_product(double local_vec1[], double local_vec2[], int local_n, MPI_Comm comm);
void Par_vector_scalar_mult(double local_vec[], double scalar, double local_result[], int local_n);

int main(void) {
    int n, local_n;
    double *local_vec1, *local_vec2;
    double scalar;
    double *local_scalar_multi1, *local_scalar_multi2;
    double dot_product;
    int comm_sz, my_rank;
    MPI_Comm comm;

    /* Initialize MPI */
    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    /* Read n and compute local n */
    Read_n(&n, &local_n, my_rank, comm_sz, comm);

    /* Allocate memory for local vectors */
    local_vec1 = malloc(local_n * sizeof(double));
    local_vec2 = malloc(local_n * sizeof(double));
    local_scalar_multi1 = malloc(local_n * sizeof(double));
    local_scalar_multi2 = malloc(local_n * sizeof(double));

    /* Read input data (vectors and scalar) */
    Read_data(local_vec1, local_vec2, &scalar, local_n, my_rank, comm_sz, comm);

    /* Print input data */
    Print_vector(local_vec1, local_n, n, "Vector 1:", my_rank, comm);
    Print_vector(local_vec2, local_n, n, "Vector 2:", my_rank, comm);
    if(my_rank == 0)
        printf("Scalar = %f\n", scalar);

    /* Compute and print dot product */
    dot_product = Par_dot_product(local_vec1, local_vec2, local_n, comm);
    if(my_rank == 0)
        printf("Dot product = %f\n", dot_product);

    /* Compute scalar multiplication and print out result */
    Par_vector_scalar_mult(local_vec1, scalar, local_scalar_multi1, local_n);
    Par_vector_scalar_mult(local_vec2, scalar, local_scalar_multi2, local_n);

    Print_vector(local_scalar_multi1, local_n, n, "Scalar * Vector 1:", my_rank, comm);
    Print_vector(local_scalar_multi2, local_n, n, "Scalar * Vector 2:", my_rank, comm);

    /* Free memory */
    free(local_scalar_multi2);
    free(local_scalar_multi1);
    free(local_vec2);
    free(local_vec1);

    MPI_Finalize();
    return 0;
}


void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;
     /* Check if all processes are okay; combine using minimum value */
    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
     /* If in the process reported an error, stop running */
    if (ok == 0) {
        int my_rank;
        MPI_Comm_rank(comm, &my_rank);
        /* Only process 0 prints the error message */
        if (my_rank == 0) {
            fprintf(stderr, "Proc %d > In %s, %s\n", my_rank, fname, message);
            fflush(stderr);
        }
        /* End MPI and exit the program */
        MPI_Finalize();
        exit(-1);
    }
}  


/* Get the input : size of the vectors, and then calculate local_n according to comm_sz */
/* where local_n is the number of elements each process obtains */
void Read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
    if(my_rank == 0){
        /* Process 0 asks the user for the total vector length */
        printf("Enter the vector length n: \n");
        scanf("%d", n_p);
    }
    /* Share the vector length with all processes */
    MPI_Bcast(n_p, 1, MPI_INT, 0, comm);
    /* Make sure n can be evenly divided among the processes */
    int local_ok = (*n_p % comm_sz == 0);
    Check_for_error(local_ok, "Read_n", "n not divisible by comm_sz", comm);
     /* Compute how many elements each process will handle */
    *local_n_p = *n_p / comm_sz;
}  


/* local_vec1 and local_vec2 store the part of the vectors that this process handles */
/* Process 0 reads the scalar and the full vectors from the user */
/* Process 0 splits vectors a and b and sends pieces to all processes */
void Read_data(double local_vec1[], double local_vec2[], double* scalar_p, int local_n, int my_rank, int comm_sz, MPI_Comm comm) {
    double* a = NULL;
    double* b = NULL;
    int n = local_n * comm_sz;

    if (my_rank == 0) {
         /* Ask the user for the scalar value */
        printf("What is the scalar?\n");
        scanf("%lf", scalar_p);
        /* Allocate memory for the full vectors */
        a = malloc(n * sizeof(double));
        b = malloc(n * sizeof(double));
        /* Ask the user to enter the first vector */
        printf("Enter the first vector (%d elements):\n", n);
        for(int i = 0; i < n; i++)
            scanf("%lf", &a[i]);
        /* Ask the user to enter the second vector */
        printf("Enter the second vector (%d elements):\n", n);
        for(int i = 0; i < n; i++)
            scanf("%lf", &b[i]);
    }
    /* Broadcast the scalar to all processes so everyone knows its value */
    MPI_Bcast(scalar_p, 1, MPI_DOUBLE, 0, comm);
    /* Split vector a into chunks and send each chunk to the corresponding process */
    MPI_Scatter(a, local_n, MPI_DOUBLE, local_vec1, local_n, MPI_DOUBLE, 0, comm);
    /* Split vector b into chunks and send each chunk to the corresponding process */
    MPI_Scatter(b, local_n, MPI_DOUBLE, local_vec2, local_n, MPI_DOUBLE, 0, comm);
    /* Free the full vectors in process 0 since each process now has its local part */
    if (my_rank == 0) {
        free(a);
        free(b);
    }
}  /* Read_data */

/* The print_vector gathers the local vectors from all processes and print the gathered vector */
void Print_vector(double local_vec[], int local_n, int n, char title[], int my_rank, MPI_Comm comm) {
    double* a = NULL;

    if (my_rank == 0) {
        /* Allocate memory to gather the full vector from all processes */
        a = malloc(n * sizeof(double));
    }
    /* Gather the  parts from all processes into the full vector in process 0 */
    MPI_Gather(local_vec, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, comm);
    if (my_rank == 0) {
        /* Print the title and the full gathered vector */
        printf("%s\n", title);
        for(int i = 0; i < n; i++)
            printf("%f ", a[i]);
        printf("\n");
        /* Free the full vector memory */
        free(a);
    }
}  


/* The function computes and returns the partial dot product of local_vec1 and local_vec2 */
double Par_dot_product(double local_vec1[], double local_vec2[], int local_n, MPI_Comm comm) {
    /* Compute the dot product of the local parts of the vectors */
    double local_sum = 0.0, global_sum = 0.0;
    for(int i = 0; i < local_n; i++)
        local_sum += local_vec1[i] * local_vec2[i];
    /* Sum all local dot products from each process to get the final global dot product */
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    /* return the dot product*/
    return global_sum;
}  



void Par_vector_scalar_mult(double local_vec[], double scalar, double local_result[], int local_n) {
     /* Multiply each element of the local vector by the scalar */
    for(int i = 0; i < local_n; i++)
        local_result[i] = scalar * local_vec[i];
}  