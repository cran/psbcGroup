

#include <stdio.h>
#include <math.h>
#include <time.h>


#include "gsl/gsl_matrix.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_sort_vector.h"

#include "gsl/gsl_eigen.h"


#include "R.h"
#include "Rmath.h"

#include "aftGL.h"


/* */
void aftGLmcmc(double survData[],
               double grpInx[],
               int *n,
               int *p,
               int *K,
               double hyperParams[],
               double startValues[],
               double *burninPerc,
               int *numReps,
               int *thin,
               double samples_alpha[],
               double samples_beta[],
               double samples_sigSq[],
               double samples_tauSq[],
               double samples_lambdaSq[],
               double samples_w[])
{
    GetRNGstate();
    
    time_t now;        
    
    
    int i, j, m, MM;
    
    
    /* Survival Data */
    
    gsl_vector *survTime    = gsl_vector_alloc(*n);
    gsl_vector *survEvent   = gsl_vector_alloc(*n);

    
    for(i = 0; i < *n; i++)
    {
        gsl_vector_set(survTime, i, survData[(0 * *n) + i]);
        gsl_vector_set(survEvent, i, survData[(1* *n) + i]);
    }
    

    int sumCen = 0;
    
    for(j = 0; j < *n; j++)
    {
        if(gsl_vector_get(survEvent, j) == 0) sumCen += 1;
    }
    
    gsl_vector *cenInx = gsl_vector_calloc(sumCen);
    
    i = 0;
    
    for(j = 0; j < *n; j++)
    {
        if(gsl_vector_get(survEvent, j) == 0)
        {
            gsl_vector_set(cenInx, i, j);
            i += 1;
        }
        
    }
    
    int nP;
    
    if(*p > 0) nP = *p;
    if(*p == 0) nP = 1;
    
    gsl_matrix *survCov = gsl_matrix_calloc(*n, nP);
        
    
    if(*p > 0)
    {
        for(i = 0; i < *n; i++)
        {
            for(j = 0; j < *p; j++) gsl_matrix_set(survCov, i, j, survData[((2+j)* *n) + i]);
        }
    }
    
    
    /* Hyperparameters */
    
    double nu0      = hyperParams[0];
    double sigSq0   = hyperParams[1];
    double alpha0   = hyperParams[2];
    double h0       = hyperParams[3];
    double rLam     = hyperParams[4];
    double deltaLam = hyperParams[5];
    

    /* Starting values */
    
    double alpha = startValues[0];
    
    gsl_vector *grp  = gsl_vector_calloc(nP);
    gsl_vector *beta = gsl_vector_calloc(nP);
    if(*p > 0)
    {
        for(j = 0; j < *p; j++)
        {
            gsl_vector_set(beta, j, startValues[1+j]);
            gsl_vector_set(grp, j, grpInx[j]);
        }
    }

    
    double sigSq = startValues[1 + *p];
    
    gsl_vector *tauSq = gsl_vector_calloc(nP);
    if(*p > 0)
    {
        for(j = 0; j < *p; j++) gsl_vector_set(tauSq, j, startValues[2+*p+j]);
    }
    
    double lambdaSq = startValues[2 + *p + *p];
    
    gsl_vector *w = gsl_vector_calloc(*n);
    for(j = 0; j < *n; j++) gsl_vector_set(w, j, startValues[3 + *p + *p + j]);


    gsl_matrix * Jmat = gsl_matrix_calloc(*n, *n);
    gsl_matrix_set_all(Jmat, 1);
    
    
    /* Variables required for storage of samples */
    
    int StoreInx;
    
    gsl_matrix *tXX = gsl_matrix_calloc(*p, *p);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, survCov, survCov, 0, tXX);
    
    gsl_matrix *invDtau = gsl_matrix_calloc(*p, *p);
    for(i = 0; i < *p; i++) gsl_matrix_set(invDtau, i, i, 1/gsl_vector_get(tauSq, i));
    gsl_matrix *Dtau = gsl_matrix_calloc(*p, *p);
    for(i = 0; i < *p; i++) gsl_matrix_set(Dtau, i, i, gsl_vector_get(tauSq, i));
    
    gsl_matrix *tXXinvD = gsl_matrix_calloc(*p, *p);
    
    gsl_vector *betaMean = gsl_vector_calloc(*p);
    gsl_matrix *betaMean_temp = gsl_matrix_calloc(*p, *n);
    gsl_matrix *betaVar = gsl_matrix_calloc(*p, *p);
    gsl_matrix *betaSample = gsl_matrix_calloc(1, *p);
    
    double meanAlp, varAlp, nuTau, invTauSq;
    double alphaSig, betaSig, shapeLam, rateLam, scaleLam;
    
    gsl_vector *wAlpXb = gsl_vector_calloc(*n);

    gsl_matrix *I_n = gsl_matrix_calloc(*n, *n);
    gsl_matrix_set_identity(I_n);
    
    
    gsl_matrix *Sigma_temp = gsl_matrix_calloc(*n, *n);
    gsl_matrix *Sigma = gsl_matrix_calloc(*n, *n);    
    gsl_matrix *XD = gsl_matrix_calloc(*n, *p);
    
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, survCov, Dtau, 0, XD);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, XD, survCov, 0, Sigma_temp);
    
    for(i = 0; i < *n; i++) gsl_matrix_set(Sigma_temp, i, i, gsl_matrix_get(Sigma_temp, i, i) + 1);
    
    gsl_matrix_add_constant(Sigma_temp, h0);
    matrixInv(Sigma_temp, Sigma);
    
    
    double eta, sample;
    int jj;
    
    gsl_vector *w_alpha = gsl_vector_calloc(*n);    
    
    int subsize = 10;
    int subsize_last = (int) *p % subsize;
    int numSub = (int) *p / subsize;
    
    int subsize_last_temp;
    
    if(subsize_last > 0) subsize_last_temp = subsize_last;
    if(subsize_last == 0) subsize_last_temp = 1;
    
    
    gsl_matrix *Xsub = gsl_matrix_calloc(*n, subsize);
    gsl_matrix *Xsub_last = gsl_matrix_calloc(*n, subsize_last_temp);

    
    gsl_vector *betasub_big = gsl_vector_calloc(subsize);
    gsl_vector *betasub_small = gsl_vector_calloc(subsize_last_temp);
    
    gsl_vector *Xbeta = gsl_vector_calloc(*n);
    gsl_vector *xbeta_sub = gsl_vector_calloc(*n);
    gsl_vector *Xbeta_xbeta_sub = gsl_vector_calloc(*n);
    
    gsl_vector *w_alpha_Xbeta = gsl_vector_calloc(*n);
    
    gsl_matrix *tXXsub = gsl_matrix_calloc(subsize, subsize);
    gsl_matrix *tXXsub_last = gsl_matrix_calloc(subsize_last_temp, subsize_last_temp);

    
    gsl_matrix *tXXinvD_sub = gsl_matrix_calloc(subsize, subsize);
    gsl_matrix *tXXinvD_sub_last = gsl_matrix_calloc(subsize_last_temp, subsize_last_temp);
    
    gsl_matrix *betaVar_sub = gsl_matrix_calloc(subsize, subsize);
    gsl_matrix *betaVar_sub_last = gsl_matrix_calloc(subsize_last_temp, subsize_last_temp);
    
    gsl_vector *betaMean_sub = gsl_vector_calloc(subsize);
    gsl_vector *betaMean_sub_last = gsl_vector_calloc(subsize_last_temp);
    
    gsl_matrix *betaMean_temp_sub = gsl_matrix_calloc(subsize, *n);
    gsl_matrix *betaMean_temp_sub_last = gsl_matrix_calloc(subsize_last_temp, *n);
    
    gsl_matrix *betaSample_sub = gsl_matrix_calloc(1, subsize);
    gsl_matrix *betaSample_sub_last = gsl_matrix_calloc(1, subsize_last_temp);
    
    gsl_vector *eval = gsl_vector_calloc(*p);
    gsl_vector *eval_sub = gsl_vector_calloc(subsize);
    gsl_vector *eval_sub_last = gsl_vector_calloc(subsize_last_temp);
    
    gsl_matrix *evec = gsl_matrix_calloc(*p, *p);
    gsl_matrix *evec_sub = gsl_matrix_calloc(subsize, subsize);
    gsl_matrix *evec_sub_last = gsl_matrix_calloc(subsize_last_temp, subsize_last_temp);
    
    gsl_eigen_symmv_workspace * work = gsl_eigen_symmv_alloc (*p);
    gsl_eigen_symmv_workspace * work_sub = gsl_eigen_symmv_alloc (subsize);
    gsl_eigen_symmv_workspace * work_sub_last = gsl_eigen_symmv_alloc (subsize_last_temp);
    
    gsl_blas_dgemv(CblasNoTrans, 1, survCov, beta, 0, Xbeta);
    
    double Move_choice, move;
    
    double p_cen = 0.1;
    
    double probVal = (double) (1-p_cen)/5;
    
    double p_alpha = probVal;
    double p_beta = probVal;
    double p_tauSq = probVal;
    double p_sigSq = probVal;
    double p_lamSq = probVal;
    
    int psd = 0;
    int kkk;
    
    double betakTbetak;
    int ll;
    int stop;
    
    
    for(MM = 0; MM < *numReps; MM++)
    {
        /*         */
        Move_choice  = runif(0, 1);
        move = 1;
        if(Move_choice > p_alpha) move = 1;
        if(Move_choice > p_alpha + p_beta) move = 1;
        if(Move_choice > p_alpha + p_beta + p_tauSq) move = 1;
        if(Move_choice > p_alpha + p_beta + p_tauSq + p_sigSq) move = 1;
        if(Move_choice > p_alpha + p_beta + p_tauSq + p_sigSq + p_lamSq) move = 2;

        
        /* Data augmentation for censored survival time            */
        
        if(move == 2)
        {
            for(i = 0; i < *n; i++)
            {
                if(gsl_vector_get(survEvent, i) == 0)
                {
                    eta = alpha + gsl_vector_get(Xbeta, i);
                    c_rtnorm(eta, sqrt(sigSq), log(gsl_vector_get(survTime, i)), 1000, 0, 1, &sample);
                    gsl_vector_set(w, i, sample);
                }
            }
        }
        
        
        /* Updating the intercept : alpha            */
        
        if(move == 1)
        {    
            meanAlp = 0;
            for(i = 0; i < *n; i++)
            {
                meanAlp += gsl_vector_get(w, i);
                meanAlp -= gsl_vector_get(Xbeta, i);
            }
            meanAlp *= h0;
            meanAlp += alpha0;
            meanAlp /= (*n*h0 + 1);
            
            varAlp = sigSq * h0 / (*n * h0 + 1);
            
            alpha = rnorm(meanAlp, sqrt(varAlp));
        }
        
        /* Updating tauSq               */   
        
        
        if(move == 1)
        {
            while(psd == 0)
            {
                psd = 1;
                
                for(i = 1; i <= *K; i++)
                {
                
                    betakTbetak = 0;
                    for(ll = 0; ll < *p; ll++)
                    {
                        if((int) gsl_vector_get(grp, ll) == i)
                        {
                            betakTbetak += pow(gsl_vector_get(beta, ll), 2);
                        }
                    }
                    nuTau = sqrt(lambdaSq * sigSq)/sqrt(betakTbetak);
                    
                    invTauSq = 0;
                    
                    while (invTauSq == 0)
                    {
                        c_rinvGauss(nuTau, lambdaSq, &invTauSq);
                    }
                    
                    for(ll = 0; ll < *p; ll++)
                    {
                        if((int) gsl_vector_get(grp, ll) == i)
                        {
                            gsl_matrix_set(invDtau, ll, ll, invTauSq);
                            gsl_vector_set(tauSq, ll, 1/invTauSq);
                            gsl_matrix_set(Dtau, ll, ll, 1/invTauSq);
                        }
                    }


                }

                
                
                /* Checking positive semi-definiteness */ 
                
                if(*p <= subsize)
                {
                    gsl_matrix_memcpy(tXXinvD, tXX);
                    gsl_matrix_add(tXXinvD, invDtau);
                    
                    matrixInv(tXXinvD, betaVar);
                    
                    gsl_eigen_symmv (betaVar, eval, evec, work);
                    
                    for(kkk = 0; kkk < *p; kkk++)
                    {
                        if(gsl_vector_get(eval, kkk) < 0)
                        {
                            psd = 0;
                        }
                    }
                    
                }
                
                if(*p > subsize)
                {   
                    for(m = 0; m < numSub; m++)
                    {
                        for(i = 0; i < *n; i++)
                        {
                            for(j = 0; j < subsize; j++) gsl_matrix_set(Xsub, i, j, gsl_matrix_get(survCov, i, (m)*subsize + j));
                        }

                        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Xsub, Xsub, 0, tXXsub);
                        
                        gsl_matrix_memcpy(tXXinvD_sub, tXXsub);
                        
                        for(j = 0; j < subsize; j++) gsl_matrix_set(tXXinvD_sub, j, j, gsl_matrix_get(tXXinvD_sub, j, j) + 1/gsl_vector_get(tauSq, (m)*subsize + j));
                        
                        
                        matrixInv(tXXinvD_sub, betaVar_sub);
                        
                        gsl_eigen_symmv (betaVar_sub, eval_sub, evec_sub, work_sub);

                        for(kkk = 0; kkk < subsize; kkk++)
                        {
                            if(gsl_vector_get(eval_sub, kkk) < 0)
                            {
                                psd = 0;
                            }
                        }
                        
                    }
                    
                    
                    if(subsize_last != 0)
                    {
                        for(i = 0; i < *n; i++)
                        {
                            for(j = 0; j < subsize_last; j++) gsl_matrix_set(Xsub_last, i, j, gsl_matrix_get(survCov, i, numSub*subsize + j));
                        }
                        
                        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Xsub_last, Xsub_last, 0, tXXsub_last);
                        
                        
                        gsl_matrix_memcpy(tXXinvD_sub_last, tXXsub_last);
                        for(j = 0; j < subsize_last; j++) gsl_matrix_set(tXXinvD_sub_last, j, j, gsl_matrix_get(tXXinvD_sub_last, j, j) + 1/gsl_vector_get(tauSq, numSub*subsize + j));
                        
                        matrixInv(tXXinvD_sub_last, betaVar_sub_last);
                        
                        gsl_eigen_symmv (betaVar_sub_last, eval_sub_last, evec_sub_last, work_sub_last);
                        
                        for(kkk = 0; kkk < subsize_last; kkk++)
                        {
                            if(gsl_vector_get(eval_sub_last, kkk) < 0)
                            {
                                psd = 0;
                            }
                        }
                        
                    }
                    
                }
                
            }
            

            
            gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, survCov, Dtau, 0, XD);
            gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, XD, survCov, 0, Sigma_temp);
            
            for(i = 0; i < *n; i++) gsl_matrix_set(Sigma_temp, i, i, gsl_matrix_get(Sigma_temp, i, i) + 1);
            
            gsl_matrix_add_constant(Sigma_temp, h0);
            matrixInv(Sigma_temp, Sigma);
        }
        
        psd = 0;
       
        /* Updating sigmaSq                    */    
        
        if(move == 1)
        {
            alphaSig = (*n + *p + nu0 + 1)/2;
            
            gsl_vector_memcpy(wAlpXb, w);
            gsl_vector_add_constant(wAlpXb, -alpha);
            gsl_blas_dgemv(CblasNoTrans, -1, survCov, beta, 1, wAlpXb);
            gsl_blas_ddot(wAlpXb, wAlpXb, &betaSig);
            
            for(i = 0; i < *p; i++)
            {
                betaSig += pow(gsl_vector_get(beta, i), 2) * gsl_matrix_get(invDtau, i, i);
            }
                        
            betaSig += nu0 * sigSq0 + pow(alpha - alpha0, 2)/h0;
            betaSig /= 2;

            c_rigamma(&sigSq, alphaSig, betaSig);
        }
        
        /* Updating lambdaSq             */          
        
        if(move == 1)
        {
            shapeLam = (*p + *K)/2 + rLam;
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
            lambdaSq = rgamma(shapeLam, scaleLam);
        }

        /* Updating regression parameters : beta                */
        
        if(move == 1)
        {
            if(*p <= subsize)
            {
                gsl_matrix_memcpy(tXXinvD, tXX);
                gsl_matrix_add(tXXinvD, invDtau);
                
                matrixInv(tXXinvD, betaVar);
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, betaVar, survCov, 0, betaMean_temp);
                
                gsl_vector_memcpy(w_alpha, w);
                gsl_vector_add_constant(w_alpha, -alpha);
                
                gsl_blas_dgemv(CblasNoTrans, 1, betaMean_temp, w_alpha, 0, betaMean);
                
                gsl_matrix_scale(betaVar, sigSq);
                
                c_rmvnorm(betaSample, betaMean, betaVar);
                
                for(i = 0; i < *p; i++) gsl_vector_set(beta, i, gsl_matrix_get(betaSample, 0, i));
                gsl_blas_dgemv(CblasNoTrans, 1, survCov, beta, 0, Xbeta);
            }
            
            if(*p > subsize)
            {
                gsl_vector_memcpy(w_alpha, w);
                gsl_vector_add_constant(w_alpha, -alpha);
                
                for(m = 0; m < numSub; m++)
                {
                    for(i = 0; i < *n; i++)
                    {
                        for(j = 0; j < subsize; j++) gsl_matrix_set(Xsub, i, j, gsl_matrix_get(survCov, i, (m)*subsize + j));
                    }
                    
                    for(j = 0; j < subsize; j++) gsl_vector_set(betasub_big, j, gsl_vector_get(beta, (m)*subsize + j));
                    
                    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Xsub, Xsub, 0, tXXsub);
                    gsl_blas_dgemv(CblasNoTrans, 1, Xsub, betasub_big, 0, xbeta_sub);
                    
                    gsl_vector_memcpy(Xbeta_xbeta_sub, Xbeta);
                    gsl_vector_sub(Xbeta_xbeta_sub, xbeta_sub);
                    gsl_vector_memcpy(w_alpha_Xbeta, w_alpha);
                    gsl_vector_sub(w_alpha_Xbeta, Xbeta_xbeta_sub);
                    
                    gsl_matrix_memcpy(tXXinvD_sub, tXXsub);
                    
                    for(j = 0; j < subsize; j++) gsl_matrix_set(tXXinvD_sub, j, j, gsl_matrix_get(tXXinvD_sub, j, j) + 1/gsl_vector_get(tauSq, (m)*subsize + j));
                    
                    matrixInv(tXXinvD_sub, betaVar_sub);
                    
                    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, betaVar_sub, Xsub, 0, betaMean_temp_sub);
                    gsl_blas_dgemv(CblasNoTrans, 1, betaMean_temp_sub, w_alpha_Xbeta, 0, betaMean_sub);
                    gsl_matrix_scale(betaVar_sub, sigSq);
                    
                    c_rmvnorm(betaSample_sub, betaMean_sub, betaVar_sub);
                    
                    for(i = 0; i < subsize; i++) gsl_vector_set(beta, (m)*subsize + i, gsl_matrix_get(betaSample_sub, 0, i));
                    gsl_blas_dgemv(CblasNoTrans, 1, survCov, beta, 0, Xbeta);
                }
                
                if(subsize_last != 0)
                {
                    for(i = 0; i < *n; i++)
                    {
                        for(j = 0; j < subsize_last; j++) gsl_matrix_set(Xsub_last, i, j, gsl_matrix_get(survCov, i, numSub*subsize + j));
                    }
                    for(j = 0; j < subsize_last; j++) gsl_vector_set(betasub_small, j, gsl_vector_get(beta, numSub*subsize + j));
                    
                    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, Xsub_last, Xsub_last, 0, tXXsub_last);
                    
                    gsl_blas_dgemv(CblasNoTrans, 1, Xsub_last, betasub_small, 0, xbeta_sub);
                    
                    gsl_vector_memcpy(Xbeta_xbeta_sub, Xbeta);
                    gsl_vector_sub(Xbeta_xbeta_sub, xbeta_sub);
                    gsl_vector_memcpy(w_alpha_Xbeta, w_alpha);
                    gsl_vector_sub(w_alpha_Xbeta, Xbeta_xbeta_sub);
                    
                    
                    gsl_matrix_memcpy(tXXinvD_sub_last, tXXsub_last);
                    for(j = 0; j < subsize_last; j++) gsl_matrix_set(tXXinvD_sub_last, j, j, gsl_matrix_get(tXXinvD_sub_last, j, j) + 1/gsl_vector_get(tauSq, numSub*subsize + j));
                    
                    matrixInv(tXXinvD_sub_last, betaVar_sub_last);
                    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, betaVar_sub_last, Xsub_last, 0, betaMean_temp_sub_last);
                    
                    gsl_blas_dgemv(CblasNoTrans, 1, betaMean_temp_sub_last, w_alpha_Xbeta, 0, betaMean_sub_last);
                    
                    gsl_matrix_scale(betaVar_sub_last, sigSq);
                    
                    c_rmvnorm(betaSample_sub_last, betaMean_sub_last, betaVar_sub_last);
                    
                    for(i = 0; i < subsize_last; i++) gsl_vector_set(beta, numSub*subsize + i, gsl_matrix_get(betaSample_sub_last, 0, i));
                    gsl_blas_dgemv(CblasNoTrans, 1, survCov, beta, 0, Xbeta);
                    
                }
                
            }
        }
        
        
        if( ( (MM+1) % *thin ) == 0 && (MM+1) > (*numReps * *burninPerc))
        {
            StoreInx = (MM+1)/(*thin)- (*numReps * *burninPerc)/(*thin);
            
            if(*p >0)
            {
                for(j = 0; j < *p; j++) samples_beta[(StoreInx - 1) * (*p) + j] = gsl_vector_get(beta, j);
            }
            
            samples_alpha[StoreInx - 1] = alpha;
            samples_sigSq[StoreInx - 1] = sigSq;
            
            samples_lambdaSq[StoreInx - 1] = lambdaSq;
            
            for(j = 0; j < *n; j++) samples_w[(StoreInx - 1) * (*n) + j] = gsl_vector_get(w, j);
        }
        
        
        if( ( (MM+1) % 5000 ) == 0)
        {
            time(&now);
            
            Rprintf("iteration: %d: %s\n", MM+1, ctime(&now));
            
            R_FlushConsole();
            R_ProcessEvents();
            
        }        
    }
    
    gsl_eigen_symmv_free (work);
    gsl_eigen_symmv_free (work_sub);
    gsl_eigen_symmv_free (work_sub_last);
    
    PutRNGstate();
    return;
}






