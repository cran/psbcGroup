

#include <stdio.h>
#include <math.h>
#include <time.h>

#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_heapsort.h"

#include "gsl/gsl_rng.h"
#include "gsl/gsl_randist.h"

#include "R.h"
#include "Rmath.h"
#include "BAFTgpLT.h"


/* */
void BAFTgpLTmcmc(double Wmat[],
                  double wUInf[],
                  double c0Inf[],
                  double Xmat[],
                  double XCmat[],
                  double grpInx[],
                  double getInx[],
                  double pk[],
                  int *n,
                  int *p,
                  int *q,
                  int *K,
                  double hyperP[],
                  double mcmcP[],
                  double startValues[],
                  int *numReps,
                  int *thin,
                  double *burninPerc,
                  double samples_w[],
                  double samples_beta[],
                  double samples_tauSq[],
                  double samples_mu[],
                  double samples_sigSq[],
                  double samples_lambdaSq[],
                  double samples_betaC[],
                  double samples_misc[])
{
    GetRNGstate();
    time_t now;
    int i, j, kk, M;
    
    const gsl_rng_type * TT;
    gsl_rng * rr;
    
    gsl_rng_env_setup();
    
    TT = gsl_rng_default;
    rr = gsl_rng_alloc(TT);
    
    /* Data */
    
    gsl_vector *wL = gsl_vector_calloc(*n);
    gsl_vector *wU = gsl_vector_calloc(*n);
    gsl_vector *wU_posinf = gsl_vector_calloc(*n);
    gsl_vector *c0 = gsl_vector_calloc(*n);
    gsl_vector *c0_neginf = gsl_vector_calloc(*n);
    
    int nP, nQ;
    
    if(*p > 0) nP = *p;
    if(*p == 0) nP = 1;
    
    gsl_matrix *X = gsl_matrix_calloc(*n, (nP));
    
    if(*q > 0) nQ = *q;
    if(*q == 0) nQ = 1;
    
    gsl_matrix *XC = gsl_matrix_calloc(*n, (nQ));
    
    for(i = 0; i < *n; i++)
    {
        gsl_vector_set(wL, i, Wmat[(0 * *n) + i]);
        gsl_vector_set(wU, i, Wmat[(1 * *n) + i]);
        gsl_vector_set(c0, i, Wmat[(2 * *n) + i]);
        gsl_vector_set(wU_posinf, i, wUInf[i]);
        gsl_vector_set(c0_neginf, i, c0Inf[i]);
        
        if(*p >0)
        {
            for(j = 0; j < *p; j++)
            {
                gsl_matrix_set(X, i, j, Xmat[(j* *n) + i]);
            }
        }
        
        if(*q >0)
        {
            for(j = 0; j < *q; j++)
            {
                gsl_matrix_set(XC, i, j, XCmat[(j* *n) + i]);
            }
        }
    }
    
    gsl_vector *grp  = gsl_vector_calloc(nP);
    gsl_vector *get_inx  = gsl_vector_calloc(nP);
    gsl_vector *p_k  = gsl_vector_calloc(nP);
    
    if(*p > 0)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_vector_set(grp, j, grpInx[j]);
        }
    }
    
    if(*K > 0)
    {
        for(j = 0; j < *K; j++)
        {
            gsl_vector_set(get_inx, j, getInx[j]);
        }
    }
    
    if(*K > 0)
    {
        for(j = 0; j < *K; j++)
        {
            gsl_vector_set(p_k, j, pk[j]);
        }
    }

    /* Hyperparameters */
    
    double a_sigSq = hyperP[0];
    double b_sigSq = hyperP[1];
    double mu0 = hyperP[2];
    double h0 = hyperP[3];
    double v = hyperP[6];
    
    /* MCMC parameters */
    
    double mu_prop_var = mcmcP[1];
    double sigSq_prop_var = mcmcP[2];
    
    int L_betaC = mcmcP[4];
    double M_betaC = mcmcP[5];
    double eps_betaC = mcmcP[6];
    
    int L_beta = mcmcP[7];
    double M_beta = mcmcP[8];
    double eps_beta = mcmcP[9];
    
    int n_betaC = 0;
    int accept_betaC = 0;
    gsl_vector *accept_betaC_100 = gsl_vector_calloc(100);
    
    int n_beta = 0;
    int accept_beta = 0;
    gsl_vector *accept_beta_100 = gsl_vector_calloc(100);
    
    /* Starting values */
    
    gsl_vector *w = gsl_vector_calloc(*n);
    gsl_vector *beta = gsl_vector_calloc(nP);
    gsl_vector *betaC = gsl_vector_calloc(nQ);
    gsl_vector *tauSq = gsl_vector_calloc(nP);
    double mu = startValues[*n+*p+*p];
    double sigSq = startValues[*n+*p+*p+1];
    double lambdaSq = startValues[*n+*p+*p+1+1];
    
    for(i = 0; i < *n; i++)
    {
        gsl_vector_set(w, i, startValues[i]);
    }
    if(*p > 0)
    {
        for(i = 0; i < *p; i++) gsl_vector_set(beta, i, startValues[*n+i]);
        for(i = 0; i < *p; i++) gsl_vector_set(tauSq, i, startValues[*n+*p+i]);
    }
    if(*q > 0)
    {
        for(i = 0; i < *q; i++) gsl_vector_set(betaC, i, startValues[*n+*p+*p+1+1+1+i]);
    }
    
    /* Variables required for storage of samples */
    
    int numSims, GG;
    
    gsl_vector *wEM = gsl_vector_calloc(*n);
    gsl_vector *betaEM = gsl_vector_calloc(nP);
    gsl_vector *betaCEM = gsl_vector_calloc(nQ);
    double muEM = 0;
    double sigSqEM = 0;
    gsl_vector *tauSqEM = gsl_vector_calloc(nP);
    
    gsl_vector *tauSqEM_sum = gsl_vector_calloc(nP);
    
    int StoreInx;
    int accept_mu = 0;
    int accept_sigSq = 0;
    
    double pW = 0.1;
    double pBeta = 0.4;
    double pTauSq = 0.1;
    double pMu = 0.1;
    double pSigSq = 0.1;
    double pLamSq = 1 - (pW + pBeta + pTauSq + pMu + pSigSq);

    gsl_vector *moveProb = gsl_vector_calloc(6);
    gsl_vector_set(moveProb, 0, pW);
    gsl_vector_set(moveProb, 1, pBeta);
    gsl_vector_set(moveProb, 2, pTauSq);
    gsl_vector_set(moveProb, 3, pMu);
    gsl_vector_set(moveProb, 4, pSigSq);
    gsl_vector_set(moveProb, 5, pLamSq);
    
    double lLH, lLHEM;
    lLH = Cal_logLH(q, c0, c0_neginf, X, XC, w, beta, betaC, mu, sigSq);

    double den;
    
    numSims = 200;
    
    for(M = 0; M < *numReps; M++)
    {
        update_w(q, wL, wU, wU_posinf, c0, X, XC, w, beta, betaC, mu, sigSq);
        lLH = Cal_logLH(q, c0, c0_neginf, X, XC, w, beta, betaC, mu, sigSq);
        
        if(*p > 0)
        {
            /* Updating beta */
            update_beta_HMC(c0, c0_neginf, X, XC, w, beta, betaC, tauSq, mu, sigSq, v, &accept_beta, accept_beta_100, &eps_beta, L_beta, M_beta, &n_beta, numReps, M, &lLH, 0);
            
            /* Updating tauSq */
            update_tauSq(K, grp, p_k, beta, tauSq, sigSq, lambdaSq);
        }
        
        if(*q > 0)
        {
            /* Updating betaC */
            update_betaC_HMC(c0, c0_neginf, X, XC, w, beta, betaC, tauSq, mu, sigSq, v, &accept_betaC, accept_betaC_100, &eps_betaC, L_betaC, M_betaC, &n_betaC, numReps, M, &lLH, 0);
        }

        update_mu(q, c0, c0_neginf, X, XC, w, beta, betaC, &mu, sigSq, mu0, h0, mu_prop_var, &accept_mu, &lLH, 0);

        update_sigSq(q, c0, c0_neginf, X, XC, w, beta, betaC, tauSq, mu, &sigSq, a_sigSq, b_sigSq, v, sigSq_prop_var, &accept_sigSq, &lLH, 0);
     
        gsl_vector_memcpy(wEM, w);
        gsl_vector_memcpy(betaEM, beta);
        gsl_vector_memcpy(betaCEM, betaC);
        muEM = mu;
        sigSqEM = sigSq;
        tauSqEM = tauSq;
        lLHEM = lLH;
        
        gsl_vector_set_zero(tauSqEM_sum);
        
        if(runif(0, 1) > 0)
        {
            for(GG = 0; GG < numSims; GG++)
            {
                update_w(q, wL, wU, wU_posinf, c0, X, XC, wEM, betaEM, betaCEM, muEM, sigSqEM);
                
                lLHEM = Cal_logLH(q, c0, c0_neginf, X, XC, wEM, betaEM, betaCEM, muEM, sigSqEM);

                update_mu(q, c0, c0_neginf, X, XC, wEM, betaEM, betaCEM, &muEM, sigSqEM, mu0, h0, mu_prop_var, &accept_mu, &lLHEM, 1);
                
                update_sigSq(q, c0, c0_neginf, X, XC, wEM, betaEM, betaCEM, tauSqEM, muEM, &sigSqEM, a_sigSq, b_sigSq, v, sigSq_prop_var, &accept_sigSq, &lLHEM, 1);
                
                if(*q > 0)
                {
                    update_betaC_HMC(c0, c0_neginf, X, XC, wEM, betaEM, betaCEM, tauSqEM, muEM, sigSqEM, v, &accept_betaC, accept_betaC_100, &eps_betaC, 2, M_betaC, &n_betaC, numReps, M, &lLHEM, 1);
                }
                
                if(*p > 0)
                {
                    update_beta_HMC(c0, c0_neginf, X, XC, wEM, betaEM, betaCEM, tauSqEM, muEM, sigSqEM, v, &accept_beta, accept_beta_100, &eps_beta, 2, M_beta, &n_beta, numReps, M, &lLHEM, 1);
                    update_tauSq(K, grp, p_k, betaEM, tauSqEM, sigSqEM, lambdaSq);
                    if(GG > (int) numSims/2)
                    {
                        gsl_vector_add(tauSqEM_sum, tauSqEM);
                    }
                }
            }
            
            den = 0;
            if(*p >0)
            {
                for(kk = 0; kk < *K; kk++)
                {
                    den += gsl_vector_get(p_k, kk) * gsl_vector_get(tauSqEM_sum, (int) gsl_vector_get(get_inx, kk)-1)/(numSims/2);
                }
            }
            
            if(*K == *p)
            {
                lambdaSq = 2**p/den;
            }else
            {
                lambdaSq = (*p+*K)/den;
            }
        }

        /* Storing posterior samples */
        
        if( ( (M+1) % *thin ) == 0 && (M+1) > (*numReps * *burninPerc))
        {
            StoreInx = (M+1)/(*thin)- (*numReps * *burninPerc)/(*thin);
            
            for(i = 0; i < *n; i++) samples_w[(StoreInx - 1) * (*n) + i] = gsl_vector_get(w, i);
            
            if(*p >0)
            {
                for(j = 0; j < *p; j++) samples_beta[(StoreInx - 1) * (*p) + j] = gsl_vector_get(beta, j);
                for(j = 0; j < *p; j++) samples_tauSq[(StoreInx - 1) * (*p) + j] = gsl_vector_get(tauSq, j);
            }
            samples_mu[StoreInx - 1] = mu;
            samples_sigSq[StoreInx - 1] = sigSq;
            samples_lambdaSq[StoreInx - 1] = lambdaSq;
            
            if(*q >0)
            {
                for(j = 0; j < *q; j++) samples_betaC[(StoreInx - 1) * (*q) + j] = gsl_vector_get(betaC, j);
            }
            
        }
        
        if(M == (*numReps - 1))
        {
            if(*p >0)
            {
                samples_misc[0] = (double) accept_beta/n_beta;
            }

            samples_misc[1] = (int) accept_mu;
            samples_misc[2] = (int) accept_sigSq;
            
            if(*q >0)
            {
                samples_misc[3] = (double) accept_betaC/n_betaC;
            }
        }
        
        if( ( (M+1) % 1000 ) == 0)
        {
            time(&now);
            Rprintf("iteration: %d: %s\n", M+1, ctime(&now));
            R_FlushConsole();
            R_ProcessEvents();
        }
    }
    
    gsl_rng_free(rr);
    
    PutRNGstate();
    return;
}





















