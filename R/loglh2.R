

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


