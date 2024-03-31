#include "../src/matrix_vector_functions.h"
#include "../src/dashsvd.h"
#include "string.h"

MKL_INT m;
MKL_INT n;

double err_res(mat_csr* A, mat* U, mat* S, mat* V, mat* S_acc)
{
    mat* SV = matrix_new(V->nrows, V->ncols);
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
    
    csr_matrix_transpose_matrix_mult(&csrA, U, SV);
    double err_max = -1;
    int i, j;
    for(i=0;i<S->nrows;i++)
    {
        double temp = 0;
        for(j=i*V->nrows;j<(i+1)*V->nrows;j++)
        {
            double temp1 = SV->d[j] - V->d[j]*S->d[i];
            temp1 = temp1*temp1;
            temp += (temp1);
        }
        temp = sqrt(temp);
        temp /= S_acc->d[i];
        if (temp > err_max) err_max = temp;
    }
    matrix_delete(SV);
    return err_max;
}

void dashSVD_test()
{
    FILE* fid;
    fid = fopen("SNAP.dat", "r");
    m = 82168;
    n = m;
    int nnz = 948464;

    mat_coo *A = coo_matrix_new(m, n, nnz);
    A->nnz = nnz;
    long long i;
    for(i=0;i<A->nnz;i++)
    {
        int ii, jj;
        double kk;
        fscanf(fid, "%d %d %lf", &ii, &jj, &kk);
        A->rows[i] = (MKL_INT)ii;
        A->cols[i] = (MKL_INT)jj;
        A->values[i] = kk;
    }
    mat_csr* D = csr_matrix_new();
    csr_init_from_coo(D, A);
    coo_matrix_delete(A);
    
    //The error metrics can be computed only when k<=100. 
	int k = 100;

    struct timeval start_timeval, end_timeval;
    
    mat* U1, *S1, *V1;
    
    gettimeofday(&start_timeval, NULL);
    
    /*This is dashSVD with default settings*/
    //dashSVD(D, &U1, &S1, &V1, k);
    
    /*This is the dashSVD with user's settings*/
    dashSVD_opt(D, &U1, &S1, &V1, k, 1000, k/2, 1e-2);
    
    gettimeofday(&end_timeval, NULL);
    
    printf("The trucated SVD of SNAP when k = %d is now computing according to dashSVD with PVE tolerance 1e-2.\n", k);

    printf("The singular values computed by dashSVD are:\n");
    
    for(i=0;i<k;i++)
        printf("%.16lf\n", S1->d[k-i-1]);
    
    printf("dashSVD comsumes %f seconds\n", get_seconds_frac(start_timeval,end_timeval));
    
    if(k>100)
    {
        printf("The error metrics can be computed only when k<=100.\n");
    }
    else
    {
        FILE *fid2 = fopen("SNAP_sv.dat", "r");
        mat* acc_s = matrix_new(k+1, 1);
        for(i=0;i<=k;i++)
            fscanf(fid2, "%lf", &acc_s->d[i]);
        
        double err_r = 0;
        err_r = err_res(D, U1, S1, V1, acc_s);
    
        sparse_status_t mkl_status;
        sparse_matrix_t csrD;
        mkl_status = mkl_sparse_s_create_csr(&csrD,
        SPARSE_INDEX_BASE_ONE,
        D->nrows,
        D->ncols,
        D->pointerI,
        &(D->pointerI[1]),
        D->cols,
        D->values);

        mkl_status = mkl_sparse_optimize(csrD);
    
        csr_matrix_transpose_matrix_mult(&csrD, U1, V1);
        mat* SS = matrix_new(k, k);
        matrix_transpose_matrix_mult(V1, V1, SS);
        double pvemax=0, pvet;
        for(i=0;i<k;i++)
        {    
            double pvet = SS->d[i*k+i]-acc_s->d[k-i-1]*acc_s->d[k-i-1];
            if(pvet < 0) pvet = -pvet;
            pvet /= (acc_s->d[k]*acc_s->d[k]);
            if(pvet>pvemax) pvemax = pvet;
        }
        double err_sigma=0;
        for(i=0;i<k;i++)
        {
            double st = (acc_s->d[i] - S1->d[k-i-1])/acc_s->d[i];
            if (st > err_sigma) err_sigma = st; 
        }
        printf("The PVE of dashSVD's Result: %e\n", pvemax);
        printf("The Residual error of dashSVD's Result: %e\n", err_r);
        printf("The Sigma error of dashSVD's Result: %e\n", err_sigma);
    }
}

int main(int argc, char const *argv[])
{
    //This is for testing dashSVD
    dashSVD_test();
    
    return 0;
}
