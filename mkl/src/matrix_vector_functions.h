#pragma once

#include <stdio.h>

#include "mkl.h"
#include "mkl_lapacke.h"
#include "mkl_vsl.h"

#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h> // for clock_gettime()


#define SEED    777
#define BRNG    VSL_BRNG_MCG31
#define METHOD  VSL_RNG_METHOD_GAUSSIAN_ICDF

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define max(x,y) (((x) > (y)) ? (x) : (y))

/* struct of the dense matrix */
typedef struct {
    MKL_INT nrows, ncols;
    double * d;
} mat;

/*struct of the vector */
typedef struct {
    MKL_INT nrows;
    double * d;
} vec;

/*struct of the sparse matrix stored in COO format*/
typedef struct {
    MKL_INT nrows, ncols;
    MKL_INT nnz; // number of non-zero element in the matrix.
    MKL_INT capacity; // number of possible nnzs.
    double *values;
    MKL_INT *rows, *cols;
} mat_coo;

/*struct of the sparse matrix stored in CSR format*/
/*typedef struct {
    MKL_INT nnz;
    MKL_INT nrows, ncols;
    double *values;
    MKL_INT *cols;
    MKL_INT *pointerB, *pointerE;
} mat_csr;*/

typedef struct {
    MKL_INT nnz;
    MKL_INT nrows, ncols;
    double *values;
    MKL_INT *cols;
    MKL_INT *pointerI;
} mat_csr;

/* initialize new matrix and set all entries to zero */
mat * matrix_new(int nrows, int ncols);

/* initialize new vector and set all entries to zero */
vec * vector_new(int nrows);

/* delete the matrix and free the space */
void matrix_delete(mat *M);

/* delete the vector and free the space */
void vector_delete(vec *v);

/* set element in column major format */
void matrix_set_element(mat *M, int row_num, int col_num, double val);

/* get element in column major format */
double matrix_get_element(mat *M, int row_num, int col_num);

/* set vector element */
void vector_set_element(vec *v, int row_num, double val);

/* get vector element */
double vector_get_element(vec *v, int row_num);

/* print to terminal */
void matrix_print(mat * M);

/* print to terminal */
void vector_print(vec * v);

/* copy contents of mat S to D  */
void matrix_copy(mat *D, mat *S);

/* subtract b from a and save result in a  */
void vector_sub(vec *a, vec *b);

/* subtract B from A and save result in A  */
void matrix_sub(mat *A, mat *B);

/* print out matrix */
void matrix_print(mat * M);

/* print out vector */
void vector_print(vec * v);

/* C = A*B ; column major */
void matrix_matrix_mult(mat *A, mat *B, mat *C);

/* C = A^T*B ; column major */
void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C);

/* C = A*B^T ; column major */
void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C);

/* extract column of a matrix into a vector */
void matrix_get_col(mat *M, int j, vec *column_vec);

/* set column of matrix to vector */
void matrix_set_col(mat *M, int j, vec *column_vec);

/* extract row i of a matrix into a vector */
void matrix_get_row(mat *M, int i, vec *row_vec);

/* put vector row_vec as row i of a matrix */
void matrix_set_row(mat *M, int i, vec *row_vec);

/* Mr = M(inds,:) */
void matrix_get_selected_rows(mat *M, int *inds, mat *Mr);

/* get seconds for recording runtime */
double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval);

/* generate Gaussian random i.i.d matrix */
void initialize_random_matrix_double(mat* M);

/* Mc = M(:, inds) */
void matrix_get_selected_columns(mat *M, int *inds, mat *Mc);

// initialize with sizes, the only interface that allocates space for coo struct
mat_coo* coo_matrix_new(int nrows, int ncols, int capacity);

/* delete the sparse matrix stored in COO */
void coo_matrix_delete(mat_coo *M);

/* print the matrix stored in COO */
void coo_matrix_print(mat_coo *M);

/* initialize the matrix stored in CSR format with a pointer, but nothing inside.*/
mat_csr* csr_matrix_new();

/* delete the sparse matrix stored in CSR */
void csr_matrix_delete(mat_csr *M);

/* print the matrix stored in CSR */
void csr_matrix_print(mat_csr *M);

/* the only interface that allocates space for mat_csr struct and initialize with M */
void csr_init_from_coo(mat_csr *D, mat_coo *M);

/* C = A*B, where A is a sparse matrix stored in CSR format*/
void csr_matrix_matrix_mult(sparse_matrix_t* csrA, mat *B, mat *C);

/* C = A'*B, where A is a sparse matrix stored in CSR format */
void csr_matrix_transpose_matrix_mult(sparse_matrix_t* csrA, mat *B, mat *C);

/*C = A'*B-d*C */
void csrAt_mult_B_minus_dC(sparse_matrix_t* csrA, mat* B, mat *C, double d);

/*C = A*B-d*C */
void csrA_mult_B_minus_dC(sparse_matrix_t* csrA, mat* B, mat *C, double d);
