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
aftGL(Y, data, grpInx, hyperParams, startValues, mcmc)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Y}{
	a data.frame containing univariate time-to-event outcomes from \code{n} subjects. It is of dimension \eqn{n\times 2}: the columns correspond to \eqn{y}, \eqn{\delta}.
}
  \item{data}{
	a data.frame containing \eqn{p} covariate vectors from \code{n} subjects. It is of dimension \eqn{n\times p}.
}
  \item{grpInx}{
a vector of \eqn{p} group indicator for each variable
}
  \item{hyperParams}{
a list containing hyperparameter values in hierarchical models:
	(\code{nu0}, \code{sigSq0}): hyperparameters for the prior of \eqn{\sigma^2}; (\code{alpha0}, \code{h0}): hyperparameters for the prior of \eqn{\alpha}; (\code{rLam}, \code{deltaLam}): hyperparameters for the prior of \eqn{\lambda^2}.
}
  \item{startValues}{
  	a list containing starting values for model parameters. See Examples below.
}
  \item{mcmc}{
	a list containing variables required for MCMC sampling. Components include,
	\code{numReps}, total number of scans; \code{thin}, extent of thinning; \code{burninPerc}, the proportion of burn-in. See Examples below.
}
}

\value{
\code{aftGL} returns an object of class \code{aftGL}. \cr}
\references{
Lee, K. H., Chakraborty, S., and Sun, J. (2017). 
Variable Selection for High-Dimensional Genomic Data with Censored Outcomes Using Group Lasso Prior. \emph{Computational Statistics and Data Analysis}, Volume 112, pages 1-13.\cr
}
\author{
Kyu Ha Lee, Sounak Chakraborty, (Tony) Jianguo Sun
}

\seealso{
\code{\link{VS}}
}

\examples{

# generate some survival data	
	set.seed(204542)
	
	p = 20
	n = 200
	logHR.true <- c(rep(4, 10), rep(0, (p-10)))	

	CovX<-matrix(0,p,p)

	for(i in 1:10){
		for(j in 1:10){
			CovX[i,j] <- 0.3^abs(i-j)
			}
		}
		
	diag(CovX) <- 1
	
	data	<- apply(rmvnorm(n, sigma=CovX, method="chol"), 2, scale)	
	pred <- as.vector(exp(rowSums(scale(data, center = FALSE, scale = 1/logHR.true))))
	
	t 		<- rexp(n, rate = pred)
	cen		<- runif(n, 0, 8)      
	tcen 		<- pmin(t, cen)
	di 		<- as.numeric(t <= cen)
	
	n <- dim(data)[1]
	p <- dim(data)[2]

	Y <- data.frame(cbind(tcen, di))
	colnames(Y) <- c("time", "event")

	grpInx <- 1:p
	K <- length(unique(grpInx))
	
	############################
	hyperParams <- list(nu0=3, sigSq0=1, alpha0=0, h0=10^6, rLam=0.5, deltaLam=2)

	############################
	startValues <- list(alpha=0.1, beta=rep(1,p), sigSq=1, tauSq=rep(0.4,p), lambdaSq=5,
	 				w=log(tcen))

	############################	
	mcmc <- list(numReps=100, thin=1, burninPerc=0.5)
	
	############################
	fit <- aftGL(Y, data, grpInx, hyperParams, startValues, mcmc)
\dontrun{  
VS(fit, X=data)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ group lasso prior }
\keyword{ ordinary lasso prior }
\keyword{ accelerated failure time models }
\keyword{ parametric models }

