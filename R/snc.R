
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





















