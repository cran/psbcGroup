setting.interval <-
function(y, delta, s, J){
	
	n <- length(y)
	
	smax	<- max(s)
	
	case0 <- which(delta == 0)
	case1 <- which(delta == 1)	
	
	case0yleq <- which(delta == 0 & y <= smax)
	case0ygeq <- which(delta == 0 & y > smax)
	case1yleq <- which(delta == 1 & y <= smax)
	case1ygeq <- which(delta == 1 & y > smax)
		

	ind.d <- ind.r <- matrix(0, n, J)

	for(i in case1yleq){
		d.mat.ind	<- min(which(s - y[i] >=0))
		ind.d[i, d.mat.ind]	<- 1
		ind.r[i, 1:d.mat.ind] <- 1		
		}
		
	for(i in case0yleq){
		cen.j <- min(which(s - y[i] >=0))		
		ind.r[i, 1:cen.j]	<- 1			
		}	
		
	if(length(union(case1ygeq, case0ygeq)) > 0){
		ind.r[union(case1ygeq, case0ygeq),]	<- 1
		}		
		
	ind.r_d	<- ind.r - ind.d;

	d	<- colSums(ind.d)
	
	list(ind.r = ind.r, ind.d = ind.d, d = d, ind.r_d = ind.r_d)
	}


bic4 <- function(t, x, alpha, beta, beta_all, sigmaSq, tauSq, nu0, sigSq0, alpha0, h0)
{
    n <- length(t)
    p <- length(p)
    p_k <- sum(beta != 0)
    sigSqM1	<- (nu0*sigSq0+(alpha-alpha0)^2/h0+sum((log(t)-alpha-as.matrix(x)%*%beta)^2)+sum(beta^2/tauSq))/(n+p_k+nu0-1)
    
    g <- length(which(beta != 0))
    val <- -2 * (loglh2(t, x, alpha, beta, sigSqM1)- loglh2(t, x, alpha, beta_all, sigmaSq)) + (p_k-p)*log(n)
    return(val)
}


loglh2 <- function(t, x, alpha, beta, sigmaSq)
{
    n <- length(t)
    lh <- 0
    xbeta <- as.vector( as.matrix(x) %*% beta)
    
    for(i in 1:n)
    {
        lh <- lh + dnorm(log(t[i]), alpha + xbeta[i], sqrt(sigmaSq), log = TRUE)
    }
    return(lh)
}




snc <- function(betaP, psi)
{
    p <- dim(betaP)[2]
    nSample <- dim(betaP)[1]
    
    betaSD <- apply(betaP, 2, sd)
    
    selected <- NULL
    for(j in 1:p)
    {
        if(sum(abs(betaP[,j]) > betaSD[j])/nSample > psi)
        {
            selected <- c(selected, j)
        }
    }
    
    if(is.null(selected))
    {
        selected <- NA
    }
    
    return(selected)
}





















