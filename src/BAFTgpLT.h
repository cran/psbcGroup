

extern void update_beta_HMC(gsl_vector *c0,
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
                            int EM);


extern void update_betaC_HMC(gsl_vector *c0,
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
                             int EM);


extern double c_min(double value1,
                    double value2);

extern double c_max(double value1,
                    double value2);


extern double logistic(double x);

extern double Cal_logLH(int *q,
                        gsl_vector *c0,
                        gsl_vector *c0_neginf,
                        gsl_matrix *X,
                        gsl_matrix *XC,
                        gsl_vector *w,
                        gsl_vector *beta,
                        gsl_vector *betaC,
                        double mu,
                        double sigSq);

extern void update_betaC(gsl_vector *c0,
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
                         int EM);

extern void update_w(int *q,
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
                     double sigSq);

extern void update_beta(int *q,
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
                        int EM);

extern void update_mu(int *q,
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
                      int EM);


extern void update_sigSq(int *q,
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
                         int EM);

extern void update_tauSq(int *K,
                         gsl_vector *grp,
                         gsl_vector *p_k,
                         gsl_vector *beta,
                         gsl_vector *tauSq,
                         double sigSq,
                         double lambdaSq);

extern void update_lambdaSq(int *K,
                            gsl_vector *grp,
                            gsl_vector *tauSq,
                            double *lambdaSq,
                            double rLam,
                            double deltaLam);

extern void matrixInv(gsl_matrix *X, gsl_matrix *Xinv);

extern void c_quadform_vMv(gsl_vector *v,
                           gsl_matrix *Minv,
                           double     *value);

extern void c_dmvnorm2(gsl_vector *x,
                       gsl_vector *mu,
                       double     sigma,
                       gsl_matrix *AInv,
                       double     *value);

extern void c_riwishart(int v,
                        gsl_matrix *X_ori,
                        gsl_matrix *sample);

extern void c_rtnorm(double mean,
                     double sd,
                     double LL,
                     double UL,
                     int LL_neginf,
                     int UL_posinf,
                     double *value);

extern void c_rttnorm(double mean,
                      double sd,
                      double LT,
                      double LL,
                      double UL,
                      int LL_neginf,
                      int UL_posinf,
                      double *value);

extern int c_multinom_sample(gsl_rng *rr,
                             gsl_vector *prob,
                             int length_prob);

extern void c_rigamma(double *temp, double alpha, double beta);

extern void c_rinvGauss(double mu, double lambda, double *val);


