\name{psbcGL}
\alias{psbcGL}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Function to Fit the Penalized Semiparametric Bayesian Cox Model with Group Lasso Prior
}
\description{
Penalized semiparametric Bayesian Cox (PSBC) model with group lasso prior is implemented to analyze survival data with high-dimensional covariates.
}
\usage{
psbcGL(survObj, priorPara, initial, rw=FALSE, mcmcPara, num.reps, 
		thin, chain = 1, save = 1000)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{survObj}{
				The list containing observed data from \code{n} subjects; 
				\code{t}, \code{di}, \code{x}
}
\item{priorPara}{
				The list containing prior parameter values; 
					\code{eta0}, \code{kappa0}, \code{c0}, \code{r}, \code{delta},
					\code{s}, \code{groupInd}
					}
  \item{initial}{
				The list containing the starting values of the parameters;
				\code{beta.ini},
				\code{lambdaSq}, \code{sigmaSq}, \code{tauSq}, \code{h}
}

\item{rw}{
When setting to "TRUE", the conventional random walk Metropolis Hastings algorithm is used. 
Otherwise, the mean and the variance of the proposal density is updated using the jumping rule described in Lee et al. (2011).
}
\item{mcmcPara}{
The list containing the values of options for Metropolis-Hastings step for \eqn{\beta};
\code{numBeta}, \code{beta.prop.var}
}
\item{num.reps}{
the number of iterations of the chain
}
  \item{thin}{
thinning
}
  \item{chain}{
the numeric name of chain in the case when running multiple chains.
}
  \item{save}{
frequency of storing the results in .Rdata file. 
For example, by setting "save = 1000", the algorithm saves the results every 1000 iterations.
}

}
\details{
\tabular{ll}{
\code{t} \tab a vector of \code{n} times to the event \cr
\code{di} \tab a vector of \code{n} censoring indicators for the event time (1=event occurred, 0=censored) \cr
\code{x} \tab covariate matrix, \code{n} observations by \code{p} variables\cr
\code{eta0} \tab scale parameter of gamma process prior for the cumulative baseline hazard, \eqn{eta0 > 0}\cr
\code{kappa0} \tab shape parameter of gamma process prior for the cumulative baseline hazard, \eqn{kappa0 > 0}\cr
\code{c0} \tab the confidence parameter of gamma process prior for the cumulative baseline hazard, \eqn{c0 > 0}\cr
\code{r} \tab the shape parameter of the gamma prior for \eqn{\lambda^2}\cr
\code{delta} \tab the rate parameter of the gamma prior for \eqn{\lambda^2}\cr
\code{s} \tab the set of time partitions for specification of the cumulative baseline hazard function\cr
\code{groupInd} \tab a vector of \code{p} group indicator for each variable\cr
\code{beta.ini} \tab the starting values for \eqn{\beta}\cr
\code{lambdaSq} \tab the starting value for \eqn{\lambda^2}\cr
\code{sigmaSq} \tab the starting value for \eqn{\sigma^2}\cr
\code{tauSq} \tab the starting values for \eqn{\tau^2}\cr
\code{h} \tab the starting values for \eqn{h}\cr
\code{numBeta} \tab the number of components in \eqn{\beta} to be updated at one iteration\cr
\code{beta.prop.var} \tab the variance of the proposal density for \eqn{\beta} when \code{rw} is set to "TRUE"\cr
}
}
\value{
\code{psbcGL} returns an object of class \code{psbcGL} \cr
\item{beta.p}{posterior samples for \eqn{\beta}}
\item{h.p}{posterior samples for \eqn{h}}
\item{tauSq.p}{posterior samples for \eqn{\tau^2}}
\item{mcmcOutcome}{The list containing posterior samples for the remaining model parameters}
}
\references{
Lee, K. H., Chakraborty, S., and Sun, J. (2011). 
Bayesian Variable Selection in Semiparametric Proportional Hazards Model for High Dimensional Survival Data. 
\emph{The International Journal of Biostatistics}, Volume 7, Issue 1, Pages 1-32. \cr

Lee, K. H., Chakraborty, S., and Sun, J. (2015). 
Survival Prediction and Variable Selection with Simultaneous Shrinkage and Grouping Priors. \emph{Statistical Analysis and Data Mining}, Volume 8, Issue 2, pages 114-127.
}
\author{
Kyu Ha Lee, Sounak Chakraborty, (Tony) Jianguo Sun
}
\note{
To fit the PSBC model with the ordinary Bayesian lasso prior (Lee et al., 2011), \code{groupInd} needs to be set to \code{1:p}. 
If the prespecified value of \code{save} is less than that of \code{num.reps}, the results are saved
as \code{.Rdata} file under the directory \code{working directory/mcmcOutcome}. 
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%objects to See Also as \code{\link{help}}, ~~~
%}
\examples{

\dontrun{

# generate some survival data
	
	set.seed(204542)
	
	p = 20
	n = 100
	beta.true <- c(rep(4, 10), rep(0, (p-10)))	

	CovX<-matrix(0,p,p)

	for(i in 1:10){
		for(j in 1:10){
			CovX[i,j] <- 0.5^abs(i-j)
			}
		}
		
	diag(CovX) <- 1
	
	survObj 	<- list()
	survObj$x	<- apply(rmvnorm(n, sigma=CovX, method="chol"), 2, scale)
	
	pred <- as.vector(exp(rowSums(scale(survObj$x, center = FALSE, scale = 1/beta.true))))
	
	t 		<- rexp(n, rate = pred)
	cen		<- runif(n, 0, 8)      
	survObj$t 		<- pmin(t, cen)
	survObj$di 		<- as.numeric(t <= cen)

	priorPara 			<- list()
	priorPara$eta0 		<- 1
	priorPara$kappa0 	<- 1
	priorPara$c0 		<- 2
	priorPara$r			<- 0.5
	priorPara$delta		<- 0.0001
	priorPara$s			<- sort(survObj$t[survObj$di == 1])
	priorPara$s			<- c(priorPara$s, 2*max(survObj$t)
	-max(survObj$t[-which(survObj$t==max(survObj$t))]))
	priorPara$J			<- length(priorPara$s)
	priorPara$groupInd	<- c(rep(1,10),2:11)

	mcmcPara				<- list()
	mcmcPara$numBeta		<- p
	mcmcPara$beta.prop.var	<- 1

	initial				<- list()
	initial$beta.ini	<- rep(0.5, p)
	initial$lambdaSq	<- 1
	initial$sigmaSq		<- runif(1, 0.1, 10)
	initial$tauSq		<- rexp(length(unique(priorPara$groupInd)),
	rate = initial$lambdaSq/2)
	initial$h			<- rgamma(priorPara$J, 1, 1)

	rw = FALSE
	num.reps = 20000
	chain = 1
	thin = 5
	save = 5

	fitGL <- psbcGL(survObj, priorPara, initial, rw=FALSE, mcmcPara, 
				num.reps, thin, chain, save)
	VS(fitGL, X=survObj$x)
				
	}		

}

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ group lasso prior }
\keyword{ ordinary lasso prior }
\keyword{ proportional hazards models }% __ONLY ONE__ keyword per line
\keyword{ semiparametric models }
