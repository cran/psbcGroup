\name{aftGL_LT}
\alias{aftGL_LT}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to Fit the Penalized Parametric Bayesian Accelerated Failure Time Model with Group Lasso Prior for Left-Truncated and Interval-Censored Data
}
\description{
Penalized parametric Bayesian accelerated failure time model with group lasso prior is implemented to analyze left-truncated and interval-censored survival data with high-dimensional covariates.
}
\usage{
aftGL_LT(Y, X, XC, grpInx, hyperParams, startValues, mcmcParams)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
Outcome matrix with three column vectors corresponding to lower and upper bounds of interval-censored data and left-truncation time
}
  \item{X}{
	Covariate matrix \eqn{p} covariate vectors from \code{n} subjects. It is of dimension \eqn{n\times p}.
}
  \item{XC}{
	Matrix for confound variables: \eqn{q} variable vectors from \code{n} subjects. It is of dimension \eqn{n\times q}.

}
  \item{grpInx}{
a vector of \eqn{p} group indicator for each variable
}
  \item{hyperParams}{
a list containing hyperparameter values in hierarchical models:
	(\code{a.sigSq}, \code{a.sigSq}): hyperparameters for the prior of \eqn{\sigma^2}; (\code{mu0}, \code{h0}): hyperparameters for the prior of \eqn{\mu}; (\code{v}): hyperparameter for the prior of \eqn{\beta_C}.
}
  \item{startValues}{
a list containing starting values for model parameters. See Examples below.
}
  \item{mcmcParams}{
	a list containing variables required for MCMC sampling. Components include,
	\code{numReps}, total number of scans; \code{thin}, extent of thinning; \code{burninPerc}, the proportion of burn-in. See Examples below.
}
}


\value{
\code{aftGL_LT} returns an object of class \code{aftGL_LT}. \cr}
\references{
Reeder, H., Haneuse, S., Lee, K. H. (2024+). 
Group Lasso Priors for Bayesian Accelerated Failure Time Models with Left-Truncated and Interval-Censored Data. \emph{under review} \cr
}
\author{
Kyu Ha Lee, Harrison Reeder
}
\seealso{
\code{\link{VS}}
}
\examples{

\dontrun{

data(survData)
X <- survData[,c(4:5)]
XC <- NULL

n <- dim(survData)[1]
p <- dim(X)[2]
q <- 0

c0 <- rep(0, n)
yL <- yU <- survData[,1]
yU[which(survData[,2] == 0)] <- Inf
Y <- cbind(yL, yU, c0)

grpInx <- 1:p
K <- length(unique(grpInx))

#####################
## Hyperparameters

a.sigSq= 0.7
b.sigSq= 0.7

mu0 <- 0
h0 <- 10^6

v = 10^6

hyperParams <- list(a.sigSq=a.sigSq, b.sigSq=b.sigSq, mu0=mu0, h0=h0, v=v)

###################
## MCMC SETTINGS

## Setting for the overall run
##
numReps    <- 100
thin    <- 1
burninPerc <- 0.5

## Tuning parameters for specific updates
##

L.beC <- 50
M.beC <- 1
eps.beC <- 0.001

L.be <- 100
M.be <- 1
eps.be <- 0.001

mu.prop.var    <- 0.5
sigSq.prop.var    <- 0.01

##

mcmcParams <- list(run=list(numReps=numReps, thin=thin, burninPerc=burninPerc),
tuning=list(mu.prop.var=mu.prop.var, sigSq.prop.var=sigSq.prop.var,
L.beC=L.beC, M.beC=M.beC, eps.beC=eps.beC,
L.be=L.be, M.be=M.be, eps.be=eps.be))

#####################
## Starting Values

w        <- log(Y[,1])
mu     <- 0.1
beta     <- rep(2, p)
sigSq    <- 0.5
tauSq <- rep(0.4, p)
lambdaSq <- 100
betaC     <- rep(0.11, q)

startValues <- list(w=w, beta=beta, tauSq=tauSq, mu=mu, sigSq=sigSq,
lambdaSq=lambdaSq, betaC=betaC)

fit <- aftGL_LT(Y, X, XC, grpInx, hyperParams, startValues, mcmcParams)

}
}
\keyword{ group lasso prior }
\keyword{ ordinary lasso prior }
\keyword{ accelerated failure time models }
\keyword{ parametric models }
