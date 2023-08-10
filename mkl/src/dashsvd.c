#include "dashsvd.h"
#include "math.h"


void eigSVD(mat* A, mat *U, mat *S, mat *V)
{   
    matrix_transpose_matrix_mult(A, A, V);
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', V->ncols, V->d, V->ncols, S->d);
    mat *V1 = matrix_new(V->ncols, V->ncols);
    matrix_copy(V1, V);
    MKL_INT i, j;
    #pragma omp parallel shared(V1,S) private(i,j) 
    {
    #pragma omp for 
        for(i=0; i<V1->ncols; i++)
        {
            S->d[i] = sqrt(S->d[i]);
            for(j=0; j<V1->nrows;j++)
            {           
                V1->d[i*V1->nrows+j] /= S->d[i];
            }
        }
    }
    mat *Uc = matrix_new(U->nrows, U->ncols);
    matrix_matrix_mult(A, V1, Uc);
    matrix_copy(U, Uc);
    matrix_delete(Uc);
    matrix_delete(V1);
}

void dashSVD(mat_csr *A, mat **U, mat **S, mat **V, int k)
{
    int s = k/2;
    int p = 1000;
    double tol = 1e-2;
    int l = k+s;
    mat *Q = matrix_new(A->nrows, l);
    mat *Qt = matrix_new(A->ncols, l);
    mat *SS = matrix_new(l, 1);
    mat *VV = matrix_new(l, l);
    
    sparse_status_t mkl_status;
    sparse_matrix_t csrA;
    mkl_status = mkl_sparse_s_create_csr(&csrA,
      SPARSE_INDEX_BASE_ONE,
      A->nrows,
      A->ncols,
      A->pointerI,
      &(A->pointerI[1]),
      A->cols,
      A->values);

    mkl_status = mkl_sparse_optimize(csrA);
    
    if (A->nrows >= A->ncols)
    {
        initialize_random_matrix_double(Q);
        csr_matrix_transpose_matrix_mult(&csrA, Q, Qt);
        eigSVD(Qt, Qt, SS, VV);
        int niter = p, i;
        double alpha = 0;
        mat* sl_now = matrix_new(l, 1);
        mat* sl = matrix_new(l, 1);
        int j;
        for(i=1;i<=niter;i++)
        {
            csr_matrix_matrix_mult(&csrA, Qt, Q);
            csrAt_mult_B_minus_dC(&csrA, Q, Qt, alpha);
            eigSVD(Qt, Qt, SS, VV);
            sl_now->d[s-1] = SS->d[s-1] + alpha;
            double pvemax=0, pvet;
            for(j=s;j<l;j++)
            {    
                sl_now->d[j] = SS->d[j] + alpha;
                double pvet = sl_now->d[j]-sl->d[j];
                if(pvet < 0) pvet = -pvet;
                pvet /= sl_now->d[s-1];
                if(pvet>pvemax) pvemax = pvet;
                sl->d[j] = sl_now->d[j];
            }
            if (pvemax < tol)  break;
            if (alpha < SS->d[0]) alpha = (alpha + SS->d[0])/2;
        }
        csr_matrix_matrix_mult(&csrA, Qt, Q);
        eigSVD(Q, Q, SS, VV);
        int inds[k]; 
        for(i=s;i<k+s;i++)
        {
            inds[i-s] = i;
        }
        *U = matrix_new(A->nrows, k);
        matrix_get_selected_columns(Q, inds, *U);
        matrix_delete(Q);
        *S = matrix_new(k, 1);
        matrix_get_selected_rows(SS, inds, *S);
        mat *VV2 = matrix_new(k+s, k);
        matrix_get_selected_columns(VV, inds, VV2);
        matrix_delete(VV);
        *V = matrix_new(A->ncols, k);
        matrix_matrix_mult(Qt, VV2, *V);
        matrix_delete(Qt);
        matrix_delete(SS);
        matrix_delete(VV2);
    }
    else
    {
        initialize_random_matrix_double(Qt);
        csr_matrix_matrix_mult(&csrA, Qt, Q);
        eigSVD(Q, Q, SS, VV);
        int niter = p, i;
        double alpha = 0;
        mat* sl_now = matrix_new(l, 1);
        mat* sl = matrix_new(l, 1);
        int j;
        for(i=1;i<=niter;i++)
        {
            csr_matrix_transpose_matrix_mult(&csrA, Q, Qt);
            csrA_mult_B_minus_dC(&csrA, Qt, Q, alpha);
            eigSVD(Q, Q, SS, VV);
            sl_now->d[s-1] = SS->d[s-1] + alpha;
            double pvemax=0, pvet;
            for(j=s;j<l;j++)
            {    
                sl_now->d[j] = SS->d[j] + alpha;
                double pvet = sl_now->d[j]-sl->d[j];
                if (pvet < 0) pvet = -pvet;
                pvet /= sl_now->d[s-1];
                if (pvet>pvemax) pvemax = pvet;
                sl->d[j] = sl_now->d[j];
            }
            if (pvemax < tol)  break;
            if (alpha < SS->d[0]) alpha = (alpha + SS->d[0])/2;
        }
        csr_matrix_transpose_matrix_mult(&csrA, Q, Qt);
        eigSVD(Qt, Qt, SS, VV);
        int inds[k]; 
        for(i=s;i<k+s;i++)
        {
            inds[i-s] = i;
        }
        *V = matrix_new(A->ncols, k);
        matrix_get_selected_columns(Qt, inds, *V);
        matrix_delete(Qt);
        *S = matrix_new(k, 1);
        matrix_get_selected_rows(SS, inds, *S);
        *U = matrix_new(A->nrows, k);
        mat *VV2 = matrix_new(k+s, k);
        matrix_get_selected_columns(VV, inds, VV2);
        matrix_delete(VV);
        matrix_matrix_mult(Q, VV2, *U);
        matrix_delete(Q);
        matrix_delete(SS);
        matrix_delete(VV2);
    }
}

void dashSVD_opt(mat_csr *A, mat **U, mat **S, mat **V, int k, int p, int s, double tol)
{
    int l = k+s;
    mat *Q = matrix_new(A->nrows, l);
    mat *Qt = matrix_new(A->ncols, l);
    mat *SS = matrix_new(l, 1);
    mat *VV = matrix_new(l, l);
    
    sparse_status_t mkl_status;
    sparse_matrix_t csrA;
    mkl_status = mkl_sparse_s_create_csr(&csrA,
      SPARSE_INDEX_BASE_ONE,
      A->nrows,
      A->ncols,
      A->pointerI,
      &(A->pointerI[1]),
      A->cols,
      A->values);

    mkl_status = mkl_sparse_optimize(csrA);
    
    if (A->nrows >= A->ncols)
    {
        initialize_random_matrix_double(Q);
        csr_matrix_transpose_matrix_mult(&csrA, Q, Qt);
        eigSVD(Qt, Qt, SS, VV);
        int niter = p, i;
        double alpha = 0;
        mat* sl_now = matrix_new(l, 1);
        mat* sl = matrix_new(l, 1);
        int j;
        for(i=1;i<=niter;i++)
        {
            csr_matrix_matrix_mult(&csrA, Qt, Q);
            csrAt_mult_B_minus_dC(&csrA, Q, Qt, alpha);
            eigSVD(Qt, Qt, SS, VV);
            sl_now->d[s-1] = SS->d[s-1] + alpha;
            double pvemax=0, pvet;
            for(j=s;j<l;j++)
            {    
                sl_now->d[j] = SS->d[j] + alpha;
                double pvet = sl_now->d[j]-sl->d[j];
                if(pvet < 0) pvet = -pvet;
                pvet /= sl_now->d[s-1];
                if(pvet>pvemax) pvemax = pvet;
                sl->d[j] = sl_now->d[j];
            }
            if (pvemax < tol)  break;
            if (alpha < SS->d[0]) alpha = (alpha + SS->d[0])/2;
        }
        csr_matrix_matrix_mult(&csrA, Qt, Q);
        eigSVD(Q, Q, SS, VV);
        int inds[k]; 
        for(i=s;i<k+s;i++)
        {
            inds[i-s] = i;
        }
        *U = matrix_new(A->nrows, k);
        matrix_get_selected_columns(Q, inds, *U);
        matrix_delete(Q);
        *S = matrix_new(k, 1);
        matrix_get_selected_rows(SS, inds, *S);
        mat *VV2 = matrix_new(k+s, k);
        matrix_get_selected_columns(VV, inds, VV2);
        matrix_delete(VV);
        *V = matrix_new(A->ncols, k);
        matrix_matrix_mult(Qt, VV2, *V);
        matrix_delete(Qt);
        matrix_delete(SS);
        matrix_delete(VV2);
    }
    else
    {
        initialize_random_matrix_double(Qt);
        csr_matrix_matrix_mult(&csrA, Qt, Q);
        eigSVD(Q, Q, SS, VV);
        int niter = p, i;
        double alpha = 0;
        mat* sl_now = matrix_new(l, 1);
        mat* sl = matrix_new(l, 1);
        int j;
        for(i=1;i<=niter;i++)
        {
            csr_matrix_transpose_matrix_mult(&csrA, Q, Qt);
            csrA_mult_B_minus_dC(&csrA, Qt, Q, alpha);
            eigSVD(Q, Q, SS, VV);
            sl_now->d[s-1] = SS->d[s-1] + alpha;
            double pvemax=0, pvet;
            for(j=s;j<l;j++)
            {    
                sl_now->d[j] = SS->d[j] + alpha;
                double pvet = sl_now->d[j]-sl->d[j];
                if (pvet < 0) pvet = -pvet;
                pvet /= sl_now->d[s-1];
                if (pvet>pvemax) pvemax = pvet;
                sl->d[j] = sl_now->d[j];
            }
            if (pvemax < tol)  break;
            if (alpha < SS->d[0]) alpha = (alpha + SS->d[0])/2;
        }
        csr_matrix_transpose_matrix_mult(&csrA, Q, Qt);
        eigSVD(Qt, Qt, SS, VV);
        int inds[k]; 
        for(i=s;i<k+s;i++)
        {
            inds[i-s] = i;
        }
        *V = matrix_new(A->ncols, k);
        matrix_get_selected_columns(Qt, inds, *V);
        matrix_delete(Qt);
        *S = matrix_new(k, 1);
        matrix_get_selected_rows(SS, inds, *S);
        *U = matrix_new(A->nrows, k);
        mat *VV2 = matrix_new(k+s, k);
        matrix_get_selected_columns(VV, inds, VV2);
        matrix_delete(VV);
        matrix_matrix_mult(Q, VV2, *U);
        matrix_delete(Q);
        matrix_delete(SS);
        matrix_delete(VV2);
    }
}

