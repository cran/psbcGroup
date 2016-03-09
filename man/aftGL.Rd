\name{aftGL}
\alias{aftGL}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to Fit the Penalized Parametric Bayesian Accelerated Failure Time Model with Group Lasso Prior
}
\description{
Penalized parametric Bayesian accelerated failure time model with group lasso prior is implemented to analyze survival data with high-dimensional covariates.
}
\usage{
aftGL(survData, grpInx, hyperParams, startValues, numReps, thin, burninPerc = 0.5)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{survData}{
	a data.frame containing the time-to-event outcome, the censoring indicator, \eqn{p} covariate vectors from \code{n} subjects. It is of dimension \eqn{n\times (p+2)}.
}
  \item{grpInx}{
a vector of \code{p} group indicator for each variable
}
  \item{hyperParams}{
a numeric vector containing hyperparameter values in hierarchical models: c(\code{nu0}, \code{sigSq0}, \code{alpha0}, \code{h0}, \code{rLam}, \code{deltaLam}).
	(\code{nu0}, \code{sigSq0}): hyperparameters for the prior of \eqn{\sigma^2}; (\code{alpha0}, \code{h0}): hyperparameters for the prior of \eqn{\alpha}; (\code{rLam}, \code{deltaLam}): hyperparameters for the prior of \eqn{\lambda^2}.
}
  \item{startValues}{
  	a numeric vector containing starting values for model parameters: c(\code{alpha}, \code{beta}, \code{sigSq}, \code{tauSq}, \code{lambdaSq}, \code{w}). See Examples below.
}
  \item{numReps}{
total number of scans
}
  \item{thin}{
extent of thinning
}
  \item{burninPerc}{
the proportion of burn-in
}
}

\value{
\code{aftGL} returns an object of class \code{aftGL}. \cr}
\references{
Lee, K. H. (2011). Bayesian Variable Selection in Parametric and Semiparametric
  High-Dimensional Survival Analysis. \emph{Ph.D. thesis}, University of
  Missouri--Columbia. \cr

Lee, K. H., Chakraborty, S., and Sun, J. 
Variable Selection for High-Dimensional Genomic Data with Censored Outcomes Using Group Lasso Prior. \emph{submitted}.
}
\author{
Kyu Ha Lee, Sounak Chakraborty, (Tony) Jianguo Sun
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\examples{

# generate some survival data	
	set.seed(204542)
	
	p = 20
	n = 100
	logHR.true <- c(rep(4, 10), rep(0, (p-10)))	

	CovX<-matrix(0,p,p)

	for(i in 1:10){
		for(j in 1:10){
			CovX[i,j] <- 0.5^abs(i-j)
			}
		}
		
	diag(CovX) <- 1
	
	x	<- apply(rmvnorm(n, sigma=CovX, method="chol"), 2, scale)	
	pred <- as.vector(exp(rowSums(scale(x, center = FALSE, scale = 1/logHR.true))))
	
	t 		<- rexp(n, rate = pred)
	cen		<- runif(n, 0, 8)      
	tcen 		<- pmin(t, cen)
	di 		<- as.numeric(t <= cen)
	
	n <- dim(x)[1]
	p <- dim(x)[2]

	survData <- data.frame(cbind(tcen, di, x))
	colnames(survData)[c(1:2)] <- c("time", "event")

	grpInx <- 1:p
	K <- length(unique(grpInx))
	
	############################
	nu0 <- 3
	sigSq0 <- 1

	alpha0 <- 0
	h0 <- 10^6

	rLam <- 0.5
	deltaLam <- 2

	hyperParams <- c(nu0, sigSq0, alpha0, h0, rLam, deltaLam)

	############################
	alpha <- 0.1
	beta <- rep(0.1, p)
	sigSq <- 1
	tauSq <- rep(0.4, p)
	lambdaSq <- 5
	w	<- log(tcen)

	startValues <- as.vector(c(alpha, beta, sigSq, tauSq, lambdaSq, w))

	############################
	numReps	<- 1000
	thin	<- 1
	burninPerc = 0.5

	fit <- aftGL(survData, grpInx, hyperParams, startValues, numReps, thin, burninPerc)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ group lasso prior }

