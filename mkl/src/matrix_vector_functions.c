/* high level matrix/vector functions using Intel MKL for blas */

#include "matrix_vector_functions.h"
#include "mkl_scalapack.h"
#include "mkl_spblas.h"


mat * matrix_new(int nrows, int ncols)
{
    mat *M = malloc(sizeof(mat));
    MKL_INT Size = nrows;
    Size *= ncols;
    M->d = (double*)calloc(Size, sizeof(double));
    M->nrows = nrows;
    M->ncols = ncols;
    return M;
}


vec * vector_new(int nrows)
{
    vec *v = malloc(sizeof(vec));
    v->d = (double*)calloc(nrows,sizeof(double));
    v->nrows = nrows;
    return v;
}


void matrix_delete(mat *M)
{
    free(M->d);
    free(M);
}


void vector_delete(vec *v)
{
    free(v->d);
    free(v);
}


void matrix_set_element(mat *M, int row_num, int col_num, double val){
    MKL_INT index = col_num;
    index *= M->nrows;
    index += row_num;
    M->d[index] = val;
}


double matrix_get_element(mat *M, int row_num, int col_num){
    MKL_INT index = col_num;
    index *= M->nrows;
    index += row_num;
    return M->d[index];
}


void vector_set_element(vec *v, int row_num, double val){
    v->d[row_num] = val;
}


double vector_get_element(vec *v, int row_num){
    return v->d[row_num];
}


void matrix_print(mat * M){
    int i,j;
    double val;
    for(i=0; i<M->nrows; i++){
        for(j=0; j<M->ncols; j++){
            val = matrix_get_element(M, i, j);
            printf("%.32f  ", val);
        }
        printf("\n");
    }
}


void vector_print(vec * v){
    int i;
    double val;
    for(i=0; i<v->nrows; i++){
        val = vector_get_element(v, i);
        printf("%f\n", val);
    }
}


void matrix_copy(mat *D, mat *S){
    MKL_INT i;
    MKL_INT length = S->nrows;
    length *= S->ncols;
    //#pragma omp parallel for
    #pragma omp parallel shared(D,S) private(i) 
    {
    #pragma omp for 
    for(i=0; i<length; i++){
        D->d[i] = S->d[i];
    }
    }
}


void matrix_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A->nrows, B->ncols, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


void matrix_transpose_matrix_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A->ncols, B->ncols, A->nrows, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


void matrix_matrix_transpose_mult(mat *A, mat *B, mat *C){
    double alpha, beta;
    alpha = 1.0; beta = 0.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A->nrows, B->nrows, A->ncols, alpha, A->d, A->nrows, B->d, B->nrows, beta, C->d, C->nrows);
}


void matrix_set_col(mat *M, int j, vec *column_vec){
    MKL_INT i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp for
    for(i=0; i<M->nrows; i++){
        matrix_set_element(M,i,j,vector_get_element(column_vec,i));
    }
    }
}


void matrix_get_col(mat *M, int j, vec *column_vec){
    MKL_INT i;
    #pragma omp parallel shared(column_vec,M,j) private(i) 
    {
    #pragma omp parallel for
    for(i=0; i<M->nrows; i++){ 
        vector_set_element(column_vec,i,matrix_get_element(M,i,j));
    }
    }
}


void matrix_get_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        vector_set_element(row_vec,j,matrix_get_element(M,i,j));
    }
    }
}


void matrix_set_row(mat *M, int i, vec *row_vec){
    int j;
    #pragma omp parallel shared(row_vec,M,i) private(j) 
    {
    #pragma omp parallel for
    for(j=0; j<M->ncols; j++){ 
        matrix_set_element(M,i,j,vector_get_element(row_vec,j));
    }
    }
}


void matrix_get_selected_rows(mat *M, int *inds, mat *Mr){
    int i;
    vec *row_vec; 
    #pragma omp parallel shared(M,Mr,inds) private(i,row_vec) 
    {
    #pragma omp parallel for
    for(i=0; i<(Mr->nrows); i++){
        row_vec = vector_new(M->ncols); 
        matrix_get_row(M,inds[i],row_vec);
        matrix_set_row(Mr,i,row_vec);
        vector_delete(row_vec);
    }
    }
}

double get_seconds_frac(struct timeval start_timeval, struct timeval end_timeval){
    long secs_used, micros_used;
    secs_used=(end_timeval.tv_sec - start_timeval.tv_sec);
    micros_used= ((secs_used*1000000) + end_timeval.tv_usec) - (start_timeval.tv_usec);
    return (micros_used/1e6); 
}


void initialize_random_matrix_double(mat *M){
    int i,m,n;
    double val;
    m = M->nrows;
    n = M->ncols;
    float a=0.0,sigma=1.0;
    long long N = m;
    N *= n;
    VSLStreamStatePtr stream;
    

    vslNewStream( &stream, BRNG,  time(NULL) );
    vdRngGaussian( METHOD, stream, N, M->d, a, sigma );
}


void matrix_get_selected_columns(mat *M, int *inds, mat *Mc){
    MKL_INT i;
    vec *col_vec = vector_new(M->nrows);
    for(i=0; i<(Mc->ncols); i++){
        matrix_get_col(M,inds[i],col_vec);
        matrix_set_col(Mc,i,col_vec);
    }
    vector_delete(col_vec);
}


mat_coo* coo_matrix_new(int nrows, int ncols, int capacity) {
    mat_coo *M = (mat_coo*)malloc(sizeof(mat_coo));
    M->values = (double*)calloc(capacity, sizeof(double));
    M->rows = (MKL_INT*)calloc(capacity, sizeof(MKL_INT));
    M->cols = (MKL_INT*)calloc(capacity, sizeof(MKL_INT));
    M->nnz = 0;
    M->nrows = nrows; M->ncols = ncols;
    M->capacity = capacity;
    return M;
}


void coo_matrix_delete(mat_coo *M) {
    free(M->values);
    free(M->cols);
    free(M->rows);
    free(M);
}


void coo_matrix_print(mat_coo *M) {
    int i;
    for (i = 0; i < M->nnz; i++) {
        printf("(%d, %d: %f), ", *(M->rows+i), *(M->cols+i), *(M->values+i));
    }
    printf("\n");
}


void csr_matrix_delete(mat_csr *M) {
    free(M->values);
    free(M->cols);
    free(M->pointerI);
    free(M);
}


void csr_matrix_print(mat_csr *M) {
    int i;
    printf("values: ");
    for (i = 0; i < M->nnz; i++) {
        printf("%f ", M->values[i]);
    }
    printf("\ncolumns: ");
    for (i = 0; i < M->nnz; i++) {
        printf("%d ", M->cols[i]);
    }
    printf("\npointerI: ");
    for (i = 0; i <= M->nrows; i++) {
        printf("%d\t", M->pointerI[i]);
    }
    printf("\n");
}


mat_csr* csr_matrix_new() {
    mat_csr *M = (mat_csr*)malloc(sizeof(mat_csr));
    return M;
}


void csr_init_from_coo(mat_csr *D, mat_coo *M) {
    D->nrows = M->nrows; 
    D->ncols = M->ncols;
    D->pointerI = (MKL_INT*)malloc((D->nrows+1)*sizeof(MKL_INT));
    D->cols = (MKL_INT*)calloc(M->nnz, sizeof(MKL_INT));
    D->nnz = M->nnz;
    
    D->values = (double*)malloc(M->nnz * sizeof(double));
    memcpy(D->values, M->values, M->nnz * sizeof(double));
    
    MKL_INT current_row, cursor=0;
    for (current_row = 0; current_row < D->nrows; current_row++) {
        D->pointerI[current_row] = cursor+1;
        while (cursor < M->nnz && M->rows[cursor]-1 == current_row) {
            D->cols[cursor] = M->cols[cursor];
            cursor++;
        }
    }
    D->pointerI[current_row]=cursor+1;
}


void csr_matrix_matrix_mult(sparse_matrix_t* csrA, mat *B, mat *C) {
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      1.,
                      *csrA,
                      descrA,
                      SPARSE_LAYOUT_COLUMN_MAJOR,
                      B->d,
                      B->ncols,   // number of right hand sides
                      B->nrows,      // ldx
                      0.,
                      C->d,
                      C->nrows);
}

void csr_matrix_transpose_matrix_mult(sparse_matrix_t* csrA, mat *B, mat *C) {
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE,
                      1.,
                      *csrA,
                      descrA,
                      SPARSE_LAYOUT_COLUMN_MAJOR,
                      B->d,
                      B->ncols,   // number of right hand sides
                      B->nrows,      // ldx
                      0.,
                      C->d,
                      C->nrows);
}

void csrAt_mult_B_minus_dC(sparse_matrix_t* csrA, mat* B, mat *C, double d) {
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_mm(SPARSE_OPERATION_TRANSPOSE,
                      1.,
                      *csrA,
                      descrA,
                      SPARSE_LAYOUT_COLUMN_MAJOR,
                      B->d,
                      B->ncols,   // number of right hand sides
                      B->nrows,      // ldx
                      -1.0*d,
                      C->d,
                      C->nrows);
}


void csrA_mult_B_minus_dC(sparse_matrix_t* csrA, mat* B, mat *C, double d) {
    struct matrix_descr descrA;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                      1.,
                      *csrA,
                      descrA,
                      SPARSE_LAYOUT_COLUMN_MAJOR,
                      B->d,
                      B->ncols,   // number of right hand sides
                      B->nrows,      // ldx
                      -1.0*d,
                      C->d,
                      C->nrows);
}

