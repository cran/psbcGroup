
bic4 <- function(t, x, alpha, beta, beta_all, sigmaSq, tauSq, nu0, sigSq0, alpha0, h0)
{
    n <- length(t)
    p <- length(beta)
    p_k <- sum(beta != 0)
    sigSqM1	<- (nu0*sigSq0+(alpha-alpha0)^2/h0+sum((log(t)-alpha-as.matrix(x)%*%beta)^2)+sum(beta^2/tauSq))/(n+p_k+nu0-1)
    
    g <- length(which(beta != 0))
    val <- -2 * (loglh2(t, x, alpha, beta, sigSqM1)- loglh2(t, x, alpha, beta_all, sigmaSq)) + (p_k-p)*log(n)
    return(val)
}

