
aftGL_LT <- function(Y,
X,
XC,
grpInx,
hyperParams,
startValues,
mcmcParams)
{
    ###
    n	<- dim(Y)[1]
    p	<- ncol(X)
    
    if(is.null(XC))
    {
    	q <- 0
    	XC <- matrix(0, n, 1)
    }else
    {
    	q	<- ncol(XC)
    }

    K <- length(unique(grpInx))
    
    ori <- grpInx
   	ori.uq <- unique(ori)
	new <- rep(NA, K)

	for(ii in 1:K)
	{
		grpInx[which(ori == ori.uq[ii])] <- ii
	}
	pk <- as.vector(table(grpInx))
	
	
	getInx <- rep(NA, K)
	for(jj in 1:K)
	{
	  getInx[jj] <- min(which(grpInx == jj))
	}
    
    ### standardize X
    X.mean <- apply(X, 2, mean)
    X.sd <- as.vector(sapply(as.data.frame(X), sd))
    
    Rinv_list <- list()
    if(p > K)
    {
        x_ortho_qr <- X
        for(i in grpInx){
          qr_temp <- qr(X[,i == grpInx])
          Rinv_list[[i]] <- solve(qr.R(qr_temp))
          x_ortho_qr[,i == grpInx] <- qr.Q(qr_temp)
        }
        X <- x_ortho_qr
    }else if(p == K)
    {
        X <- (X - matrix(rep(X.mean, each = n), n, p))/matrix(rep(X.sd, each = n), n, p)
    }
	
    ###
    hyperP  <- as.vector(c(hyperParams$a.sigSq, hyperParams$b.sigSq, hyperParams$mu0, hyperParams$h0, 1, 1, hyperParams$v))
    mcmcP   <- as.vector(c(1, mcmcParams$tuning$mu.prop.var, mcmcParams$tuning$sigSq.prop.var, 1, mcmcParams$tuning$L.beC, mcmcParams$tuning$M.beC, mcmcParams$tuning$eps.beC, mcmcParams$tuning$L.be, mcmcParams$tuning$M.be, mcmcParams$tuning$eps.be))
    startVal <- as.vector(c(startValues$w, startValues$beta, startValues$tauSq, startValues$mu, startValues$sigSq, startValues$lambdaSq, startValues$betaC))
    
    numReps     <- mcmcParams$run$numReps
    thin        <- mcmcParams$run$thin
    burninPerc  <- mcmcParams$run$burninPerc
    nStore      <- numReps/thin * (1 - burninPerc)
    
    W <- Y
    W[,1] <- log(Y[,1])
    W[,2] <- log(Y[,2])
    W[,3] <- log(Y[,3])
    
    for(i in 1:n) 
    {
      if(W[i,1] == -Inf)
      {
        W[i,1] <- -9.9e10
      }
    }
    
    wUInf <- rep(0, n)
    for(i in 1:n) if(W[i,2] == Inf)
    {
        W[i,2] <- 9.9e10
        wUInf[i] <- 1
    }
    
    c0Inf <- rep(0, n)
    for(i in 1:n) if(W[i,3] == -Inf)
    {
        W[i,3] <- -9.9e10
        c0Inf[i] <- 1
    }
    
    mcmcRet     <- .C("BAFTgpLTmcmc",
    Wmat            = as.double(as.matrix(W)),
    wUInf			= as.double(wUInf),
    c0Inf			= as.double(c0Inf),
    Xmat           	= as.double(as.matrix(X)),
    XCmat               = as.double(as.matrix(XC)),
    grpInx            = as.double(grpInx),
    getInx            = as.double(getInx),
	pk            = as.double(pk),
    n				= as.integer(n),
    p				= as.integer(p),
    q				= as.integer(q),
    K                = as.integer(K),
    hyperP          = as.double(hyperP),
    mcmcP           = as.double(mcmcP),
    startValues 		= as.double(startVal),
    numReps			= as.integer(numReps),
    thin				= as.integer(thin),
    burninPerc      = as.double(burninPerc),
    samples_w       = as.double(rep(0, nStore*n)),
    samples_beta    = as.double(rep(0, nStore*p)),
    samples_tauSq    = as.double(rep(0, nStore*p)),
    samples_mu   = as.double(rep(0, nStore*1)),
    samples_sigSq   = as.double(rep(0, nStore*1)),
    samples_lambdaSq   = as.double(rep(0, nStore*1)),
    samples_betaC    = as.double(rep(0, nStore*q)),
    samples_misc    = as.double(rep(0, 1+1+1+1)))
    
    w.p <- matrix(as.vector(mcmcRet$samples_w), nrow=nStore, byrow=T)
    if(p >0)
    {
        if(p > K)
        {
            betaScaled.p <- matrix(as.vector(mcmcRet$samples_beta), nrow=nStore, byrow=T)
            beta.p <- betaScaled.p
            for(kk in 1:dim(betaScaled.p)[1])
            {
                for(i in ori)
                {
                    beta.p[kk, i == ori] <- Rinv_list[[i]] %*% betaScaled.p[kk, i == ori]
                }
            }
        }else if(p == K)
        {
            betaScaled.p <- matrix(as.vector(mcmcRet$samples_beta), nrow=nStore, byrow=T)
            beta.p <- matrix(as.vector(mcmcRet$samples_beta), nrow=nStore, byrow=T)/ matrix(rep(X.sd, each = nStore), nStore, p)
        }
        
    }else
    {
        beta.p <- NULL
    }
    
    if(p > 0){
        tauSq.p     <- matrix(mcmcRet$samples_tauSq, nrow = nStore, byrow = TRUE)
    }
    if(p == 0){
        tauSq.p     <- NULL
    }
    
    mu.p <- matrix(as.vector(mcmcRet$samples_mu), nrow=nStore, byrow=T)
    sigSq.p <- matrix(as.vector(mcmcRet$samples_sigSq), nrow=nStore, byrow=T)
    lambdaSq.p <- matrix(as.vector(mcmcRet$samples_lambdaSq), nrow=nStore, byrow=T)
    
    if(q > 0)
    {
        betaC.p <- matrix(as.vector(mcmcRet$samples_betaC), nrow=nStore, byrow=T)
    }else
    {
        betaC.p <- NULL
    }
    
    if(p >0)
    {
        accept.beta	 <- as.vector(mcmcRet$samples_misc[1])
    }else
    {
        accept.beta <- NULL
    }
    
    accept.mu	 <- as.vector(mcmcRet$samples_misc[2])
    accept.sigSq	 <- as.vector(mcmcRet$samples_misc[3])
    
    if(q >0)
    {
        accept.betaC	 <- mcmcRet$samples_misc[4]
    }else
    {
        accept.betaC <- NULL
    }
    
    ret <- list(w.p = w.p, betaScaled.p = betaScaled.p, beta.p = beta.p, betaC.p=betaC.p, tauSq.p=tauSq.p, mu.p=mu.p, sigSq.p = sigSq.p, lambdaSq.p=lambdaSq.p, accept.beta = accept.beta, accept.mu = accept.mu, accept.sigSq = accept.sigSq, accept.betaC=accept.betaC, getInx=ori, getInxNew=getInx, Rinv_list=Rinv_list, X.mean=X.mean, X.s=X.sd, hyperParams=hyperParams, mcmcParams=mcmcParams)
    
    class(ret) <- "aftGL_LT"
    return(ret)
}

