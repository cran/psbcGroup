                                                                                                                                                                                                     #include <stdio.h>
#include <math.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_sf.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include "R.h"
#include "Rmath.h"

#include "BAFTgpLT.h"




/* Updating betaC using HMC*/

void update_betaC_HMC(gsl_vector *c0,
                      gsl_vector *c0_neginf,
                      gsl_matrix *X,
                      gsl_matrix *XC,
                      gsl_vector *w,
                      gsl_vector *beta,
                      gsl_vector *betaC,
                      gsl_vector *tauSq,
                      double mu,
                      double sigSq,
                      double v,
                      int *accept_betaC,
                      gsl_vector *accept_betaC_100,
                      double *eps_betaC,
                      int L_betaC,
                      double M_betaC,
                      int *n_betaC,
                      int *numReps,
                      int M,
                      double *lLH,
                      int EM)
{
    int i, j, l, u;
    
    double tempC, tempC_prop, val, val_prop, sumAccept, eta, eta_prop;
    
    double U_star, U_prop, K_star, K_prop, logR;
    int accept_ind;
    
    int n = XC -> size1;
    int q = XC -> size2;

    gsl_vector *xbeta = gsl_vector_calloc(n);
    gsl_vector *xbetaC = gsl_vector_calloc(n);
    gsl_vector *xbetaC_prop = gsl_vector_calloc(n);
    gsl_vector *a_vec = gsl_vector_calloc(n);
    gsl_vector *a_vec_prop = gsl_vector_calloc(n);
    
    gsl_vector *betaC_ini = gsl_vector_calloc(q);
    gsl_vector *betaC_star = gsl_vector_calloc(q);
    gsl_vector *betaC_prop = gsl_vector_calloc(q);
    gsl_vector *p_ini = gsl_vector_calloc(q);
    gsl_vector *p_star = gsl_vector_calloc(q);
    gsl_vector *p_prop = gsl_vector_calloc(q);
    gsl_vector *Delta_star = gsl_vector_calloc(q);
    gsl_vector *Delta_prop = gsl_vector_calloc(q);
    
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    gsl_blas_dgemv(CblasNoTrans, 1, XC, betaC, 0, xbetaC);
    
    accept_ind = 0;
    if(EM == 0)
    {
        *n_betaC += 1;
    }

    gsl_vector_memcpy(betaC_ini, betaC);
    gsl_vector_memcpy(betaC_star, betaC);
    
    for(j = 0; j < q; j++)
    {
        gsl_vector_set(p_star, j, rnorm(0, sqrt(M_betaC)));
    }

    for(i = 0; i < n; i++)
    {
        eta = mu + gsl_vector_get(xbeta, i) + gsl_vector_get(xbetaC, i);
        val = (gsl_vector_get(w, i) - eta)/sigSq;

        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            tempC = (eta - gsl_vector_get(c0, i))/sqrt(sigSq);
            val -= exp(dnorm(tempC, 0, 1, 1)-pnorm(tempC, 0, 1, 1, 1))/sqrt(sigSq);
        }
        gsl_vector_set(a_vec, i, val);
    }
    
    U_star = -*lLH;
    
    gsl_blas_dgemv(CblasTrans, 1, XC, a_vec, 0, Delta_star);

    for(j = 0; j < q; j++)
    {
        U_star -= dnorm(gsl_vector_get(betaC, j)/v, 0, 1, 1);
        gsl_vector_set(Delta_star, j, gsl_vector_get(Delta_star, j) - gsl_vector_get(betaC, j)/pow(v, 2));
    }
    
    gsl_vector_memcpy(p_ini, Delta_star);
    gsl_vector_scale(p_ini, 0.5* *eps_betaC);
    gsl_vector_add(p_ini, p_star);
    
    U_prop = 0;
    for(l = 1; l <= L_betaC; l++)
    {
        gsl_vector_memcpy(betaC_prop, p_ini);
        gsl_vector_scale(betaC_prop, *eps_betaC/M_betaC);
        gsl_vector_add(betaC_prop, betaC_ini);

        gsl_blas_dgemv(CblasNoTrans, 1, XC, betaC_prop, 0, xbetaC_prop);
        
        gsl_vector_set_zero(Delta_prop);
        
        for(i = 0; i < n; i++)
        {
            eta_prop = mu + gsl_vector_get(xbeta, i) + gsl_vector_get(xbetaC_prop, i);
            val_prop = (gsl_vector_get(w, i) - eta_prop)/sigSq;
            
            if(gsl_vector_get(c0_neginf, i) == 0)
            {
                tempC_prop = (eta_prop - gsl_vector_get(c0, i))/sqrt(sigSq);
                val_prop -= exp(dnorm(tempC_prop, 0, 1, 1)-pnorm(tempC_prop, 0, 1, 1, 1))/sqrt(sigSq);
                if(l == L_betaC)
                {
                    U_prop -= dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1) - pnorm(gsl_vector_get(c0, i), eta_prop, sqrt(sigSq), 0, 1);
                }
            }else
            {
                if(l == L_betaC)
                {
                    U_prop -= dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1);
                }
            }
            gsl_vector_set(a_vec_prop, i, val_prop);
        }
        
        gsl_blas_dgemv(CblasTrans, 1, XC, a_vec_prop, 0, Delta_prop);
        
        for(j = 0; j < q; j++)
        {
            if(l == L_betaC)
            {
                U_prop -= dnorm(gsl_vector_get(betaC_prop, j)/v, 0, 1, 1);
            }
            gsl_vector_set(Delta_prop, j, gsl_vector_get(Delta_prop, j) - gsl_vector_get(betaC_prop, j)/pow(v, 2));
        }
 
        if(l < L_betaC)
        {
            gsl_vector_memcpy(p_prop, Delta_prop);
            gsl_vector_scale(p_prop, *eps_betaC);
            gsl_vector_add(p_prop, p_ini);
            
            gsl_vector_memcpy(betaC_ini, betaC_prop);
            gsl_vector_memcpy(p_ini, p_prop);
        }else if(l == L_betaC)
        {
            gsl_vector_memcpy(p_prop, Delta_prop);
            gsl_vector_scale(p_prop, 0.5**eps_betaC);
            gsl_vector_add(p_prop, p_ini);
        }
        
    }
    
    K_star = 0;
    K_prop = 0;
    for(j = 0; j < q; j++)
    {
        K_star += pow(gsl_vector_get(p_star, j), 2);
        K_prop += pow(gsl_vector_get(p_prop, j), 2);
    }
    
    K_star *= 0.5/M_betaC;
    K_prop *= 0.5/M_betaC;
    
    logR = -(U_prop + K_prop) + U_star + K_star;
    
    u = log(runif(0, 1)) < logR;

    if(u == 1)
    {
        gsl_vector_memcpy(betaC, betaC_prop);
        *lLH = -U_prop;
        
        for(j = 0; j < q; j++)
        {
            *lLH -= dnorm(gsl_vector_get(betaC, j)/v, 0, 1, 1);
        }
        if(EM == 0)
        {
            *accept_betaC += 1;
        }
    }

    if(M < (int) *numReps/3 && EM == 0)
    {
        if(*n_betaC <= 100)
        {
            accept_ind = *n_betaC-1;
            gsl_vector_set(accept_betaC_100, accept_ind, u);
        }else if(*n_betaC > 100)
        {
            accept_ind = (int) *n_betaC % 10 + 90 - 1;
            gsl_vector_set(accept_betaC_100, accept_ind, u);
        }
        
        if((int) *n_betaC % 10 == 0 && (int) *n_betaC >= 100)
        {
            sumAccept = 0;
            for(i = 0; i < 99; i++)
            {
                sumAccept += gsl_vector_get(accept_betaC_100, i);
            }
            if(sumAccept / 100 < 0.60)
            {
                *eps_betaC = *eps_betaC * 0.9;
            }else if(sumAccept / 100 > 0.70)
            {
                *eps_betaC = *eps_betaC * 1.1;
            }
            accept_ind = 90;
            
            for(i = 0; i < 90; i++)
            {
                gsl_vector_set(accept_betaC_100, i, gsl_vector_get(accept_betaC_100, i+10));
            }
            for(i = 90; i < 99; i++)
            {
                gsl_vector_set(accept_betaC_100, i, 0);
            }
        }
    }

    gsl_vector_free(xbeta);
    gsl_vector_free(xbetaC);
    gsl_vector_free(xbetaC_prop);
    gsl_vector_free(a_vec);
    gsl_vector_free(a_vec_prop);
    
    gsl_vector_free(betaC_ini);
    gsl_vector_free(betaC_star);
    gsl_vector_free(betaC_prop);
    gsl_vector_free(p_ini);
    gsl_vector_free(p_star);
    gsl_vector_free(p_prop);
    gsl_vector_free(Delta_star);
    gsl_vector_free(Delta_prop);
    
    return;
}







/* Updating beta using HMC*/

void update_beta_HMC(gsl_vector *c0,
                      gsl_vector *c0_neginf,
                      gsl_matrix *X,
                      gsl_matrix *XC,
                      gsl_vector *w,
                      gsl_vector *beta,
                      gsl_vector *betaC,
                      gsl_vector *tauSq,
                      double mu,
                      double sigSq,
                      double v,
                      int *accept_beta,
                      gsl_vector *accept_beta_100,
                      double *eps_beta,
                      int L_beta,
                      double M_beta,
                      int *n_beta,
                      int *numReps,
                      int M,
                      double *lLH,
                      int EM)
{
    int i, j, l, u;
    
    double tempC, tempC_prop, val, val_prop, sumAccept, eta, eta_prop;
    
    double U_star, U_prop, K_star, K_prop, logR;
    int accept_ind;
    
    int n = X -> size1;
    int p = X -> size2;
    
    gsl_vector *xbetaC = gsl_vector_calloc(n);
    gsl_vector *xbeta = gsl_vector_calloc(n);
    gsl_vector *xbeta_prop = gsl_vector_calloc(n);
    gsl_vector *a_vec = gsl_vector_calloc(n);
    gsl_vector *a_vec_prop = gsl_vector_calloc(n);
    
    gsl_vector *beta_ini = gsl_vector_calloc(p);
    gsl_vector *beta_star = gsl_vector_calloc(p);
    gsl_vector *beta_prop = gsl_vector_calloc(p);
    gsl_vector *p_ini = gsl_vector_calloc(p);
    gsl_vector *p_star = gsl_vector_calloc(p);
    gsl_vector *p_prop = gsl_vector_calloc(p);
    gsl_vector *Delta_star = gsl_vector_calloc(p);
    gsl_vector *Delta_prop = gsl_vector_calloc(p);
    
    gsl_blas_dgemv(CblasNoTrans, 1, XC, betaC, 0, xbetaC);
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    
    accept_ind = 0;
    if(EM == 0)
    {
        *n_beta += 1;
    }
    
    gsl_vector_memcpy(beta_ini, beta);
    gsl_vector_memcpy(beta_star, beta);
    
    for(j = 0; j < p; j++)
    {
        gsl_vector_set(p_star, j, rnorm(0, sqrt(M_beta)));
    }
    
    for(i = 0; i < n; i++)
    {
        eta = mu + gsl_vector_get(xbeta, i) + gsl_vector_get(xbetaC, i);
        val = (gsl_vector_get(w, i) - eta)/sigSq;
        
        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            tempC = (eta - gsl_vector_get(c0, i))/sqrt(sigSq);
            val -= exp(dnorm(tempC, 0, 1, 1)-pnorm(tempC, 0, 1, 1, 1))/sqrt(sigSq);
        }
        gsl_vector_set(a_vec, i, val);
    }
    
    U_star = -*lLH;
    
    gsl_blas_dgemv(CblasTrans, 1, X, a_vec, 0, Delta_star);
    
    for(j = 0; j < p; j++)
    {
        U_star -= dnorm(gsl_vector_get(beta, j)/pow(gsl_vector_get(tauSq, j)*sigSq, 0.5), 0, 1, 1);
        gsl_vector_set(Delta_star, j, gsl_vector_get(Delta_star, j) - gsl_vector_get(beta, j)/gsl_vector_get(tauSq, j)/sigSq);
    }
    
    gsl_vector_memcpy(p_ini, Delta_star);
    gsl_vector_scale(p_ini, 0.5* *eps_beta);
    gsl_vector_add(p_ini, p_star);
    
    U_prop = 0;
    for(l = 1; l <= L_beta; l++)
    {
        gsl_vector_memcpy(beta_prop, p_ini);
        gsl_vector_scale(beta_prop, *eps_beta/M_beta);
        gsl_vector_add(beta_prop, beta_ini);
        
        gsl_blas_dgemv(CblasNoTrans, 1, X, beta_prop, 0, xbeta_prop);
        
        gsl_vector_set_zero(Delta_prop);
        
        for(i = 0; i < n; i++)
        {
            eta_prop = mu + gsl_vector_get(xbeta_prop, i) + gsl_vector_get(xbetaC, i);
            val_prop = (gsl_vector_get(w, i) - eta_prop)/sigSq;
            
            if(gsl_vector_get(c0_neginf, i) == 0)
            {
                tempC_prop = (eta_prop - gsl_vector_get(c0, i))/sqrt(sigSq);
                val_prop -= exp(dnorm(tempC_prop, 0, 1, 1)-pnorm(tempC_prop, 0, 1, 1, 1))/sqrt(sigSq);
                if(l == L_beta)
                {
                    U_prop -= dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1) - pnorm(gsl_vector_get(c0, i), eta_prop, sqrt(sigSq), 0, 1);
                }
            }else
            {
                if(l == L_beta)
                {
                    U_prop -= dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1);
                }
            }
            gsl_vector_set(a_vec_prop, i, val_prop);
        }
        
        gsl_blas_dgemv(CblasTrans, 1, X, a_vec_prop, 0, Delta_prop);
        
        for(j = 0; j < p; j++)
        {
            if(l == L_beta)
            {
                U_prop -= dnorm(gsl_vector_get(beta_prop, j)/pow(gsl_vector_get(tauSq, j)*sigSq, 0.5), 0, 1, 1);
            }
            gsl_vector_set(Delta_prop, j, gsl_vector_get(Delta_prop, j) - gsl_vector_get(beta_prop, j)/gsl_vector_get(tauSq, j)/sigSq);
        }
 
        if(l < L_beta)
        {
            gsl_vector_memcpy(p_prop, Delta_prop);
            gsl_vector_scale(p_prop, *eps_beta);
            gsl_vector_add(p_prop, p_ini);
            
            gsl_vector_memcpy(beta_ini, beta_prop);
            gsl_vector_memcpy(p_ini, p_prop);
        }else if(l == L_beta)
        {
            gsl_vector_memcpy(p_prop, Delta_prop);
            gsl_vector_scale(p_prop, 0.5**eps_beta);
            gsl_vector_add(p_prop, p_ini);
        }
    }
    
    K_star = 0;
    K_prop = 0;
    for(j = 0; j < p; j++)
    {
        K_star += pow(gsl_vector_get(p_star, j), 2);
        K_prop += pow(gsl_vector_get(p_prop, j), 2);
    }
    
    K_star *= 0.5/M_beta;
    K_prop *= 0.5/M_beta;
    
    logR = -(U_prop + K_prop) + U_star + K_star;
    
    u = log(runif(0, 1)) < logR;
    

    if(u == 1)
    {
        gsl_vector_memcpy(beta, beta_prop);
 
        *lLH = -U_prop;
        for(j = 0; j < p; j++)
        {
            *lLH -= dnorm(gsl_vector_get(beta, j)/pow(gsl_vector_get(tauSq, j)*sigSq, 0.5), 0, 1, 1);
        }
        if(EM == 0)
        {
            *accept_beta += 1;
        }
    }
    
    if(M < (int) *numReps/3 && EM == 0)
    {
        if(*n_beta <= 100)
        {
            accept_ind = *n_beta-1;
            gsl_vector_set(accept_beta_100, accept_ind, u);
        }else if(*n_beta > 100)
        {
            accept_ind = (int) *n_beta % 10 + 90 - 1;
            gsl_vector_set(accept_beta_100, accept_ind, u);
        }
        
        if((int) *n_beta % 10 == 0 && (int) *n_beta >= 100)
        {
            sumAccept = 0;
            for(i = 0; i < 99; i++)
            {
                sumAccept += gsl_vector_get(accept_beta_100, i);
            }
            if(sumAccept / 100 < 0.60)
            {
                *eps_beta = *eps_beta * 0.9;
            }else if(sumAccept / 100 > 0.70)
            {
                *eps_beta = *eps_beta * 1.1;
            }
            accept_ind = 90;
            
            for(i = 0; i < 90; i++)
            {
                gsl_vector_set(accept_beta_100, i, gsl_vector_get(accept_beta_100, i+10));
            }
            for(i = 90; i < 99; i++)
            {
                gsl_vector_set(accept_beta_100, i, 0);
            }
            
        }
    }
    
    gsl_vector_free(xbetaC);
    gsl_vector_free(xbeta);
    gsl_vector_free(xbeta_prop);
    gsl_vector_free(a_vec);
    gsl_vector_free(a_vec_prop);
    
    gsl_vector_free(beta_ini);
    gsl_vector_free(beta_star);
    gsl_vector_free(beta_prop);
    gsl_vector_free(p_ini);
    gsl_vector_free(p_star);
    gsl_vector_free(p_prop);
    gsl_vector_free(Delta_star);
    gsl_vector_free(Delta_prop);
    
    return;
}



/* Updating betaC */

void update_betaC(gsl_vector *c0,
                  gsl_vector *c0_neginf,
                  gsl_matrix *X,
                  gsl_matrix *XC,
                  gsl_vector *w,
                  gsl_vector *beta,
                  gsl_vector *betaC,
                  gsl_vector *tauSq,
                  double mu,
                  double sigSq,
                  double v,
                  double betaC_prop_var,
                  gsl_vector *accept_betaC,
                  double *lLH,
                  int EM)
{
    int i, j, u;
    double eta_prop, loglh_prop, logprior, logprior_prop, logR;
    
    int n = XC -> size1;
    int q = XC -> size2;
    
    gsl_vector *xbeta = gsl_vector_calloc(n);
    
    gsl_vector *betaC_prop = gsl_vector_calloc(q);
    gsl_vector *xbetaC_prop = gsl_vector_calloc(n);
    
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    
    j = (int) runif(0, q);
    
    loglh_prop = 0;
    
    gsl_vector_memcpy(betaC_prop, betaC);
    gsl_vector_set(betaC_prop, j, rnorm(gsl_vector_get(betaC, j), sqrt(betaC_prop_var)));
    gsl_blas_dgemv(CblasNoTrans, 1, XC, betaC_prop, 0, xbetaC_prop);
    
    for(i = 0; i < n; i++)
    {
        eta_prop = mu + gsl_vector_get(xbeta, i) +gsl_vector_get(xbetaC_prop, i);
        
        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1) - pnorm(gsl_vector_get(c0, i), eta_prop, sqrt(sigSq), 0, 1);
        }else
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1);
        }
    }
    
    logprior = dnorm(gsl_vector_get(betaC, j), 0, v, 1);
    logprior_prop = dnorm(gsl_vector_get(betaC_prop, j), 0, v, 1);
    
    logR = loglh_prop - *lLH + logprior_prop - logprior;
    u = log(runif(0, 1)) < logR;

    if(u == 1)
    {
        gsl_vector_memcpy(betaC, betaC_prop);
        if(EM == 0)
        {
            gsl_vector_set(accept_betaC, j, gsl_vector_get(accept_betaC, j) + 1);
            *lLH = loglh_prop;
        }
    }
    
    gsl_vector_free(xbeta);
    gsl_vector_free(betaC_prop);
    gsl_vector_free(xbetaC_prop);
    return;
}






/* Updating w */

void update_w(int *q,
              gsl_vector *wL,
              gsl_vector *wU,
              gsl_vector *wU_posinf,
              gsl_vector *c0,
              gsl_matrix *X,
              gsl_matrix *XC,
              gsl_vector *w,
              gsl_vector *beta,
              gsl_vector *betaC,
              double mu,
              double sigSq)
{
    double eta, sample;
    int i;
    int n = w -> size;
    double xbetaC;
    
    gsl_vector *xbeta = gsl_vector_calloc(n);
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    
    for(i=0;i<n;i++)
    {

        if(gsl_vector_get(wU, i) != gsl_vector_get(wL, i))
        {
    
            eta = mu+gsl_vector_get(xbeta, i);
            
            if(*q > 0)
            {
                gsl_vector_view XCrow = gsl_matrix_row(XC, i);
                gsl_blas_ddot(&XCrow.vector, betaC, &xbetaC);
                eta += xbetaC;
            }
            
            c_rtnorm(eta, sqrt(sigSq), gsl_vector_get(wL, i), gsl_vector_get(wU, i), 0, gsl_vector_get(wU_posinf, i), &sample);
            gsl_vector_set(w, i, sample);
        }
        else if(gsl_vector_get(wU, i) == gsl_vector_get(wL, i))
        {
  
            gsl_vector_set(w, i, gsl_vector_get(wU, i));
        }
    }
    gsl_vector_free(xbeta);
    return;
}


/* log-likelihood */

double Cal_logLH(int *q,
                 gsl_vector *c0,
                 gsl_vector *c0_neginf,
                 gsl_matrix *X,
                 gsl_matrix *XC,
                 gsl_vector *w,
                 gsl_vector *beta,
                 gsl_vector *betaC,
                 double mu,
                 double sigSq)
{
    int i;
    double eta, loglh;
    double xbetaC;
    int n = X -> size1;
    
    gsl_vector *xbeta = gsl_vector_calloc(n);
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    
    loglh = 0;
    
    for(i = 0; i < n; i++)
    {
        eta = mu + gsl_vector_get(xbeta, i);
        if(*q > 0)
        {
            gsl_vector_view XCrow = gsl_matrix_row(XC, i);
            gsl_blas_ddot(&XCrow.vector, betaC, &xbetaC);
            eta += xbetaC;
        }
        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            loglh += dnorm(gsl_vector_get(w, i), eta, sqrt(sigSq), 1) - pnorm(gsl_vector_get(c0, i), eta, sqrt(sigSq), 0, 1);
        }else
        {
            loglh += dnorm(gsl_vector_get(w, i), eta, sqrt(sigSq), 1);
        }
    }

    gsl_vector_free(xbeta);
    return loglh;
}


/* Updating beta */

void update_beta(int *q,
                 gsl_vector *c0,
                 gsl_vector *c0_neginf,
                 gsl_matrix *X,
                 gsl_matrix *XC,
                 gsl_vector *w,
                 gsl_vector *beta,
                 gsl_vector *betaC,
                 gsl_vector *tauSq,
                 double mu,
                 double sigSq,
                 double beta_prop_var,
                 gsl_vector *accept_beta,
                 double *lLH,
                 int EM)
{
    int i, j, u;
    double eta_prop, loglh_prop, logprior, logprior_prop, logR;
    double xbetaC;
    
    int n = X -> size1;
    int p = X -> size2;
    
    gsl_vector *beta_prop = gsl_vector_calloc(p);
    gsl_vector *xbeta_prop = gsl_vector_calloc(n);
    
    j = (int) runif(0, p);
    
    loglh_prop = 0;
    
    gsl_vector_memcpy(beta_prop, beta);
    gsl_vector_set(beta_prop, j, rnorm(gsl_vector_get(beta, j), sqrt(beta_prop_var)));
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta_prop, 0, xbeta_prop);
    
    for(i = 0; i < n; i++)
    {
        eta_prop = mu + gsl_vector_get(xbeta_prop, i);
        
        if(*q > 0)
        {
            gsl_vector_view XCrow = gsl_matrix_row(XC, i);
            gsl_blas_ddot(&XCrow.vector, betaC, &xbetaC);
            eta_prop += xbetaC;
        }

        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1) - pnorm(gsl_vector_get(c0, i), eta_prop, sqrt(sigSq), 0, 1);
        }else
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1);
        }
    }
    
    logprior = dnorm(gsl_vector_get(beta, j), 0, sqrt(gsl_vector_get(tauSq, j)*sigSq), 1);
    logprior_prop = dnorm(gsl_vector_get(beta_prop, j), 0, sqrt(gsl_vector_get(tauSq, j)*sigSq), 1);
    
    logR = loglh_prop - *lLH + logprior_prop - logprior;
    u = log(runif(0, 1)) < logR;
    
    if(u == 1)
    {
        gsl_vector_memcpy(beta, beta_prop);
        if(EM == 0)
        {
            gsl_vector_set(accept_beta, j, gsl_vector_get(accept_beta, j) + 1);
            *lLH = loglh_prop;
        }
    }
    
    
    gsl_vector_free(beta_prop);
    gsl_vector_free(xbeta_prop);
    return;
}



/* Updating mu */

void update_mu(int *q,
               gsl_vector *c0,
               gsl_vector *c0_neginf,
               gsl_matrix *X,
               gsl_matrix *XC,
               gsl_vector *w,
               gsl_vector *beta,
               gsl_vector *betaC,
               double *mu,
               double sigSq,
               double mu0,
               double h0,
               double mu_prop_var,
               int *accept_mu,
               double *lLH,
               int EM)
{
    int i, u;
    double eta_prop, loglh_prop, logR, mu_prop, logprior, logprior_prop;
    double xbetaC;
    
    int n = X -> size1;
    
    gsl_vector *xbeta = gsl_vector_calloc(n);
    
    loglh_prop = 0;
    mu_prop = rnorm(*mu, sqrt(mu_prop_var));
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    
    for(i = 0; i < n; i++)
    {
        eta_prop = mu_prop + gsl_vector_get(xbeta, i);
        if(*q > 0)
        {
            gsl_vector_view XCrow = gsl_matrix_row(XC, i);
            gsl_blas_ddot(&XCrow.vector, betaC, &xbetaC);
            eta_prop += xbetaC;
        }
        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1) - pnorm(gsl_vector_get(c0, i), eta_prop, sqrt(sigSq), 0, 1);
        }else
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta_prop, sqrt(sigSq), 1);
        }
    }
    
    logprior = dnorm(*mu, mu0, sqrt(h0), 1);
    logprior_prop = dnorm(mu_prop, mu0, sqrt(h0), 1);
    
    logR = loglh_prop - *lLH + logprior_prop - logprior;
    u = log(runif(0, 1)) < logR;
    if(u == 1)
    {
        *mu = mu_prop;
        *lLH = loglh_prop;
        if(EM == 0)
        {
            *accept_mu += 1;
        }
    }
    
    gsl_vector_free(xbeta);
    return;
}












/* Updating sigmaSq */

void update_sigSq(int *q,
                  gsl_vector *c0,
                  gsl_vector *c0_neginf,
                  gsl_matrix *X,
                  gsl_matrix *XC,
                  gsl_vector *w,
                  gsl_vector *beta,
                  gsl_vector *betaC,
                  gsl_vector *tauSq,
                  double mu,
                  double *sigSq,
                  double a_sigSq,
                  double b_sigSq,
                  double v,
                  double sigSq_prop_var,
                  int *accept_sigSq,
                  double *lLH,
                  int EM)
{
    int i, j, u;
    double eta, loglh_prop, logR, gamma_prop, sigSq_prop;
    double logprior, logprior_prop, logprior1, logprior1_prop;
    double xbetaC;
    
    int n = X -> size1;
    int p = X -> size2;
    gsl_vector *xbeta = gsl_vector_calloc(n);
    
    loglh_prop = 0;
    gamma_prop = rnorm(log(*sigSq), sqrt(sigSq_prop_var));
    sigSq_prop = exp(gamma_prop);
    gsl_blas_dgemv(CblasNoTrans, 1, X, beta, 0, xbeta);
    
    for(i = 0; i < n; i++)
    {
        eta = mu + gsl_vector_get(xbeta, i);
        
        if(*q > 0)
        {
            gsl_vector_view XCrow = gsl_matrix_row(XC, i);
            gsl_blas_ddot(&XCrow.vector, betaC, &xbetaC);
            eta += xbetaC;
        }
        
        if(gsl_vector_get(c0_neginf, i) == 0)
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta, sqrt(sigSq_prop), 1) - pnorm(gsl_vector_get(c0, i), eta, sqrt(sigSq_prop), 0, 1);
        }else
        {
            loglh_prop += dnorm(gsl_vector_get(w, i), eta, sqrt(sigSq_prop), 1);
        }
    }
    
    logprior = (-a_sigSq-1)*log(*sigSq)-b_sigSq /(*sigSq);
    logprior_prop = (-a_sigSq-1)*log(sigSq_prop)-b_sigSq/sigSq_prop;
    
    logprior1 = 0;
    logprior1_prop = 0;
    
    for(j = 0; j < p; j++)
    {
        logprior1 += dnorm(gsl_vector_get(beta, j), 0, sqrt(gsl_vector_get(tauSq, j)* *sigSq), 1);
        logprior1_prop += dnorm(gsl_vector_get(beta, j), 0, sqrt(gsl_vector_get(tauSq, j)* sigSq_prop), 1);
    }
    
    logR = loglh_prop - *lLH + logprior_prop - logprior + logprior1_prop - logprior1 + gamma_prop - log(*sigSq);
    
    u = log(runif(0, 1)) < logR;
    
    if(u == 1)
    {
        *sigSq = sigSq_prop;
        *lLH = loglh_prop;
        if(EM == 0)
        {
            *accept_sigSq += 1;
        }
    }
    
    gsl_vector_free(xbeta);
    return;
}






/* Updating tauSq */

void update_tauSq(int *K,
                  gsl_vector *grp,
                  gsl_vector *p_k,
                  gsl_vector *beta,
                  gsl_vector *tauSq,
                  double sigSq,
                  double lambdaSq)
{
    int i, ll;
    double betakTbetak, nuTau, invTauSq;
    
    int p = beta -> size;
    
    for(i = 1; i <= *K; i++)
    {
        betakTbetak = 0;
        for(ll = 0; ll < p; ll++)
        {
            if((int) gsl_vector_get(grp, ll) == i)
            {
                betakTbetak += pow(gsl_vector_get(beta, ll), 2);
            }
        }
        nuTau = sqrt(lambdaSq * sigSq * gsl_vector_get(p_k, i-1))/sqrt(betakTbetak);
        
        invTauSq = 0;
        
        while (invTauSq == 0)
        {
            c_rinvGauss(nuTau, lambdaSq * gsl_vector_get(p_k, i-1), &invTauSq);
        }
        
        for(ll = 0; ll < p; ll++)
        {
            if((int) gsl_vector_get(grp, ll) == i)
            {
                gsl_vector_set(tauSq, ll, 1/invTauSq);
            }
        }
    }
    
    return;
}







void update_lambdaSq(int *K,
                     gsl_vector *grp,
                     gsl_vector *tauSq,
                     double *lambdaSq,
                     double rLam,
                     double deltaLam)
{
    int i, jj, stop;
    double shapeLam, rateLam, scaleLam;
    
    int p = tauSq -> size;
    
    shapeLam = (p + *K)/2 + rLam;
    rateLam = 0;
    
    for(i = 1; i <= *K; i++)
    {
        stop = 0;
        jj = 0;
        while (stop == 0)
        {
            if((int) gsl_vector_get(grp, jj) == i)
            {
                rateLam += gsl_vector_get(tauSq, jj);
                stop = 1;
            }
            jj += 1;
        }
    }
    
    rateLam /= 2;
    rateLam += deltaLam;
    scaleLam = 1/rateLam;

    *lambdaSq = rgamma(shapeLam, scaleLam);
    
    return;
}











