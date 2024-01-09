

####
## Variable Selection
####
##

VS <- function(fit, X, psiVec=seq(0.001, 1, 0.001))
{
    p <- dim(fit$beta.p)[2]
    beta.me <- apply(fit$beta.p, 2, mean)
    
    selList <- list()
    nSel <- rep(NA, length(psiVec))
    
    for(i in 1:length(psiVec))
    {
        if(i %% 1000 == 0){
            cat("SNC: ", i, "out of ", length(psiVec), sep = " ", fill = T)
        }
        
        selList[[i]] <- snc(fit$beta.p, psiVec[i])
        nSel[i] <- length(selList[[i]])
        if(is.na(selList[[i]][1]))
        {
            nSel[i] <- 0
        }
    }
    
    bicVec <- rep(NA, length(psiVec))
    
    if(inherits(fit, "aftGL"))
    {
        w.me <- apply(fit$w.p, 2, mean)
        alpha.me <- mean(fit$alpha.p)
        sigSq.me <- mean(fit$sigSq.p)
        tauSq.me <- apply(fit$tauSq.p, 2, mean)
        
        nu0 <- fit$hyperParams$nu0
        sigSq0 <- fit$hyperParams$sigSq0
        alpha0 <- fit$hyperParams$alpha0
        h0 <- fit$hyperParams$h0
                
        for(i in 1:length(psiVec))
        {
            if(i %% 100 == 0){
                cat("BIC: i = ", i, "out of ", length(psiVec), sep = " ", fill = T)
            }
            betaHat <- beta.me
            
            ind0 <- c(1:p)[-selList[[i]]]
            betaHat[ind0] <- 0
            if(is.na(ind0[1]) & length(ind0) >0)
            {
                betaHat <- rep(0, p)
            }
            
            gam <- coef(lm(w.me ~ as.matrix(X)%*%betaHat))[2]
            if(is.na(gam))
            {
                gam = 1
            }
            
            bicVec[i] <- bic4(t=exp(w.me), x=X, alpha=alpha.me, beta=gam*betaHat, beta_all = beta.me, sigmaSq=sigSq.me, tauSq=tauSq.me, nu0, sigSq0, alpha0, h0)
        }
        
    }else if(inherits(fit, "psbcEN") | inherits(fit, "psbcFL")| inherits(fit, "psbcGL"))
    {
        for(i in 1:length(psiVec))
        {
            if(i %% 100 == 0){
                cat("BIC: i = ", i, "out of ", length(psiVec), sep = " ", fill = T)
            }
            betaHat <- beta.me
            
            ind0 <- c(1:p)[-selList[[i]]]
            betaHat[ind0] <- 0
            if(is.na(ind0[1]) & length(ind0) >0)
            {
                betaHat <- rep(0, p)
            }
            
            gam <- as.vector(coef(coxph(Surv(fit$t, fit$di) ~ as.matrix(X)%*%betaHat)))
            
            if(is.na(gam))
            {
                gam = 1
            }
            
            a.star <- coxph(Surv(fit$t, fit$di) ~ as.matrix(X)%*%(betaHat*gam))
            a.all <- coxph(Surv(fit$t, fit$di) ~ as.matrix(X)%*%beta.me)
            
            bicVec[i] <- -2 * (a.star$loglik[2] - a.all$loglik[2])+(sum(betaHat != 0)-p)*log(length(fit$t))
        }
    }
    
    vSel <- list(selList = selList, nSel = nSel, bicVec = bicVec, psiVec = psiVec)
    Sel.ind <- selList[[which(bicVec==min(bicVec))[1]]]
    
    ret <- list(Sel.ind=Sel.ind, vSel=vSel)
    
    ##
    cat("\nIndicators for variables selected based on SNC-BIC thresholding\n")
    print(Sel.ind)
    
    return(ret)
}


