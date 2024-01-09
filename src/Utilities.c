


#include <stdio.h>
#include <math.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_heapsort.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include "R.h"
#include "Rmath.h"

#include "aftGL.h"
#include "BAFTgpLT.h"

#define Pi 3.141592653589793238462643383280

/*
 Minimum of two numbers
 */
double c_min(double value1,
             double value2)
{
    double min = (value1 <= value2) ? value1 : value2;
    return min;
}



/*
 Maximum of two numbers
 */
double c_max(double value1,
             double value2)
{
    double max = (value1 >= value2) ? value1 : value2;
    return max;
}


int c_multinom_sample(gsl_rng *rr,
                      gsl_vector *prob,
                      int length_prob)
{
    int ii, val;
    int KK = length_prob;
    double probK[KK];
    
    for(ii = 0; ii < KK; ii++)
    {
        probK[ii] = gsl_vector_get(prob, ii);
    }
    
    unsigned int samples[KK];
    
    gsl_ran_multinomial(rr, KK, 1, probK, samples);
    
    for(ii = 0; ii < KK; ii++)
    {
        if(samples[ii] == 1) val = ii + 1;
    }
    
    return val;
}

double logistic(double x)
{
    double value = exp(x)/(1+exp(x));
    return value;
}

/*
 Random generation from truncated normal distribution
 */
void c_rtnorm(double mean,
              double sd,
              double LL,
              double UL,
              int LL_neginf,
              int UL_posinf,
              double *value)
{
    int caseNO;
    int stop=0;
    double y, a, z, u, val, rho;
    
    LL = (LL-mean)/sd;
    UL = (UL-mean)/sd;
    
    if( (LL < 0 && UL_posinf == 1) || (LL_neginf == 1 && UL >0) || (UL_posinf == 0 && LL_neginf == 0 && LL <0 && UL > 0  && (UL-LL) > sqrt(2* Pi)))
    {
        caseNO = 1;
    }else if(LL >= 0 && UL > LL + 2*sqrt(exp(1))/(LL+sqrt(pow(LL,2)+4))*exp((2*LL-LL*sqrt(pow(LL, 2)+4))/4))
    {
        caseNO = 2;
    }else if(UL <= 0 && -LL > -UL + 2*sqrt(exp(1))/(-UL+sqrt(pow(UL, 2)+4))*exp((2*UL+UL*sqrt(pow(UL, 2)+4))/4))
    {
        caseNO = 3;
    }else
    {
        caseNO = 4;
    }
    
    
    if(caseNO == 1)
    {
        while(stop == 0)
        {
            y = rnorm(0, 1);
            if(y > LL && y < UL)
            {
                stop = 1;
                val = y;
            }
        }
    }
    if(caseNO == 2)
    {
        while(stop == 0)
        {
            a = (LL + sqrt(pow(LL, 2)+4))/2;
            z = rexp((double) 1/a) + LL;
            u = runif(0, 1);
            if(u <= exp(-pow(z-a, 2)/2) && z <= UL)
            {
                stop = 1;
                val = z;
            }
        }
    }
    if(caseNO == 3)
    {
        while(stop == 0)
        {
            a = (-UL + sqrt(pow(UL, 2)+4))/2;
            z = rexp((double) 1/a) - UL;
            u = runif(0, 1);
            if(u <= exp(-pow(z-a, 2)/2) && z <= -LL)
            {
                stop = 1;
                val = -z;
            }
        }
    }
    if(caseNO == 4)
    {
        while(stop == 0)
        {
            z = runif(LL, UL);
            if(LL >0)
            {
                rho = exp((pow(LL, 2) - pow(z, 2))/2);
            }else if(UL <0)
            {
                rho = exp((pow(UL, 2) - pow(z, 2))/2);
            }else
            {
                rho = exp(- pow(z, 2)/2);
            }
            u = runif(0, 1);
            
            if(u <= rho)
            {
                stop = 1;
                val = z;
            }
        }
    }
    *value = mean + val * sd;
    return;
}





/*
 Random generation from truncated-truncated normal distribution
 */
void c_rttnorm(double mean,
               double sd,
               double LT,
               double LL,
              double UL,
              int LL_neginf,
              int UL_posinf,
              double *value)
{
    int stop=0;
    double sample_tnorm;

    
    if(UL_posinf == 1)
    {
        while(stop == 0)
        {
            c_rtnorm(mean, sd, LT, UL, 0, 1, &sample_tnorm);
            if(sample_tnorm > LL) stop = 1;
        }
    }else
    {
        while(stop == 0)
        {
            c_rtnorm(mean, sd, LT, UL, 0, 1, &sample_tnorm);
            if(sample_tnorm > LL && sample_tnorm < UL) stop = 1;
        }
    }


    *value = sample_tnorm;
    return;
}





/*
 Random number generation for multivariate normal distribution
 mean (n)
 Var (n x n)
 sample (numSpl x n)
 */


void c_rmvnorm(gsl_matrix *sample,
               gsl_vector *mean,
               gsl_matrix *Var)
{
    gsl_matrix_set_zero(sample);
    
    int n = sample->size2;
    int numSpl = sample->size1;
    int i, j;
    double spl;
    
    gsl_matrix *temp = gsl_matrix_alloc(n, n);
    
    gsl_matrix_memcpy(temp, Var);
    gsl_linalg_cholesky_decomp(temp);
    
    for(i = 0; i < n; i ++){
        for(j = 0; j < n; j++){
            if(i > j){
                gsl_matrix_set(temp, i, j, 0);
            }
        }
    }
    
    for(i = 0; i < numSpl; i ++){
        for(j = 0; j < n; j ++){
            spl = rnorm(0, 1);
            gsl_matrix_set(sample, i, j, spl);
        }
    }
    
    gsl_blas_dtrmm(CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, 1, temp, sample);
    
    for(i = 0; i < numSpl; i++){
        gsl_vector_view sampleRow = gsl_matrix_row(sample, i);
        gsl_vector_add(&sampleRow.vector, mean);
    }
    
    gsl_matrix_free(temp);
    
    return;
}






/*
 Random number generation for inverse gamma distribution
 alpha = shape, beta = rate
 
 */
void c_rigamma(double *temp,
               double alpha,
               double beta)
{
    double shape = alpha;
    double scale = (double) 1/beta;
    double gam = 1;
    
    if(alpha > 0 && beta > 0){
        gam = rgamma(shape, scale);
    }
    *temp = (double) 1/gam;
    return;
}





/*
 Random number generation for inverse Gaussian distribution
 
 */

void c_rinvGauss(double nu,
               double lambda,
               double *val)
{
	double b=nu/lambda/2;
	double a=nu*b;
	double c=4*nu*lambda;
	double d=pow(nu,2);
    
    *val = rnorm(0,1);
    
    double u=unif_rand();
    double v=*val * *val;
    double x=nu+a*v-b*sqrt(c*v+d*v*v);
    *val=(u<(nu/(nu+x)))?x:d/x;
    if (*val<0.0)
    {
        v=x;
    }
	PutRNGstate();
}	




/*
 Evaluate the inverse of the matrix X
 */
void matrixInv(gsl_matrix *X, gsl_matrix *Xinv)
{
    int signum;
    int d = X->size1;
    gsl_matrix      *XLU = gsl_matrix_calloc(d, d);
    gsl_permutation *p   = gsl_permutation_alloc(d);
    
    gsl_matrix_memcpy(XLU, X);
    gsl_linalg_LU_decomp(XLU, p, &signum);
    gsl_linalg_LU_invert(XLU, p, Xinv);
    
    gsl_matrix_free(XLU);
    gsl_permutation_free(p);
    return;
}


/*
 Evaluate the quadratic form: v^T M^{-1} v
 */
void c_quadform_vMv(gsl_vector *v,
                    gsl_matrix *Minv,
                    double     *value)
{
    int    d = v->size;
    gsl_vector *tempVec = gsl_vector_calloc(d);
    
    gsl_blas_dsymv(CblasUpper, 1, Minv, v, 0, tempVec);
    gsl_blas_ddot(v, tempVec, value);
    
    gsl_vector_free(tempVec);
    return;
}

/*
 Density calculation for a multivariate normal distribution
 */
void c_dmvnorm2(gsl_vector *x,
                gsl_vector *mu,
                double     sigma,
                gsl_matrix *AInv,
                double     *value)
{
    int signum, K = x->size;
    double sigmaSqInv = pow(sigma, -2);
    
    gsl_vector *diff       = gsl_vector_alloc(K);
    gsl_matrix *SigmaInv   = gsl_matrix_alloc(K, K);
    gsl_matrix *SigmaInvLU = gsl_matrix_alloc(K, K);
    gsl_permutation *p     = gsl_permutation_alloc(K);
    
    gsl_vector_memcpy(diff, x);
    gsl_vector_sub(diff, mu);
    
    gsl_matrix_memcpy(SigmaInv, AInv);
    gsl_matrix_scale(SigmaInv, sigmaSqInv);
    gsl_matrix_memcpy(SigmaInvLU, SigmaInv);
    gsl_linalg_LU_decomp(SigmaInvLU, p, &signum);
    
    c_quadform_vMv(diff, SigmaInv, value);
    
    *value = (log(gsl_linalg_LU_det(SigmaInvLU, signum)) - log(pow(2*Pi, K)) - *value) / 2;
    
    gsl_vector_free(diff);
    gsl_matrix_free(SigmaInv);
    gsl_matrix_free(SigmaInvLU);
    gsl_permutation_free(p);
    return;
}



/*
 Random generation from the Inverse Wishart distribution
 */

void c_riwishart(int v,
                 gsl_matrix *X_ori,
                 gsl_matrix *sample)
{
    int i, j, df;
    double normVal;
    int p = X_ori->size1;
    
    gsl_matrix *X = gsl_matrix_calloc(p, p);
    matrixInv(X_ori, X);
    
    gsl_matrix *cholX = gsl_matrix_calloc(p, p);
    gsl_matrix *ZZ = gsl_matrix_calloc(p, p);
    gsl_matrix *XX = gsl_matrix_calloc(p, p);
    gsl_matrix *KK = gsl_matrix_calloc(p, p);
    
    gsl_matrix_memcpy(cholX, X);
    gsl_linalg_cholesky_decomp(cholX);
    
    for(i = 0; i < p; i ++)
    {
        for(j = 0; j < i; j ++)
        {
            gsl_matrix_set(cholX, i, j, 0);
        }
    }
    
    for(i = 0; i < p; i++)
    {
        df = v - i;
        gsl_matrix_set(ZZ, i, i, sqrt(rchisq(df)));
    }
    
    for(i = 0; i < p; i++)
    {
        for(j = 0; j < i; j ++)
        {
            normVal = rnorm(0, 1);
            gsl_matrix_set(ZZ, i, j, normVal);
        }
    }
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, ZZ, cholX, 0, XX);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, XX, XX, 0, KK);
    matrixInv(KK, sample);
    
    gsl_matrix_free(X);
    gsl_matrix_free(cholX);
    gsl_matrix_free(XX);
    gsl_matrix_free(ZZ);
    gsl_matrix_free(KK);
    
}





/*
 Calculating column sums of matrix X
 */
void c_colSums(gsl_matrix *X, gsl_vector *v)
{
    int numCol = X->size2;
    int numRow = X->size1;    
    int i, j;
    double sum = 0;
    for(j = 0; j < numCol; j++)
    {
        i = 0;
        while(i < numRow)
        {
            sum = sum + gsl_matrix_get(X, i, j);
            i++;
        }
        gsl_vector_set(v, j, sum);
        sum = 0;
    }
    return;
}


/*
 Calculating row sums of matrix X
 */
void c_rowSums(gsl_matrix *X, gsl_vector *v)
{
    int numCol = X->size2;
    int numRow = X->size1;    
    int i, j;
    double sum = 0;
    for(i = 0; i < numRow; i++)
    {
        j = 0;
        while(j < numCol)
        {
            sum = sum + gsl_matrix_get(X, i, j);
            j++;
        }
        gsl_vector_set(v, i, sum);
        sum = 0;
    }
    return;
}


/*
 Replicate a vector v into rows of a matrix X
 */
void c_repVec_Rowmat(gsl_vector *v, gsl_matrix *X)
{
    int length = v->size;
    int numRep = X->size1;
    int i, j;
    for(i = 0; i < numRep; i++)
    {
        for(j = 0; j < length; j++)
        {
            gsl_matrix_set(X, i, j, gsl_vector_get(v, j));
        }
    }
    return;
}



/*
 Replicate a vector v into columns of a matrix X
 */
void c_repVec_Colmat(gsl_vector *v, gsl_matrix *X)
{
    int length = v->size;
    int numRep = X->size2;
    int i, j;
    for(j = 0; j < numRep; j++)
    {
        for(i = 0; i < length; i++)
        {
            gsl_matrix_set(X, i, j, gsl_vector_get(v, i));
        }
    }
    return;
}

















