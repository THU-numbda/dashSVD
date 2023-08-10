#pragma once

#include "mkl.h"
#include "matrix_vector_functions.h"

/*[U, S, V] = eigSVD(A) for m >> n, using eig(A'*A)*/
void eigSVD(mat* A, mat *U, mat *S, mat *V);

/*[U, S, V] = dashSVD(A, k) with default settings*/
void dashSVD(mat_csr *A, mat **U, mat **S, mat **V, int k);

/*[U, S, V] = dashSVD(A, k, p, s, tol) with user's settings*/
void dashSVD_opt(mat_csr *A, mat **U, mat **S, mat **V, int k, int p, int s, double tol);
