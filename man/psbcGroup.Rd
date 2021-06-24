\name{psbcGroup}
\alias{psbcGroup}
\docType{package}
\title{
Penalized Parametric and Semiparametric Bayesian Survival Models with Shrinkage and Grouping Priors
}
\description{
The package provides algorithms for fitting penalized parametric and semiparametric Bayesian survival models with elastic net, fused lasso, and group lasso priors.
}
\details{
The package includes following functions:
\tabular{ll}{
\code{psbcEN} \tab The function to fit the PSBC model with elastic net prior \cr
\code{psbcFL} \tab The function to fit the PSBC model with fused lasso prior \cr
\code{psbcGL} \tab The function to fit the PSBC model with group lasso or Bayesian lasso prior \cr
\code{aftGL} \tab The function to fit the parametric accelerated failure time model with group lasso \cr
}
\tabular{ll}{
Package: \tab psbcGroup\cr
Type: \tab Package\cr
Version: \tab 1.5\cr
Date: \tab 2021-06-23\cr
License: \tab GPL (>= 2)\cr
LazyLoad: \tab yes\cr
}
%~~ An overview of how to use the package, including the most ~~
%~~ important functions ~~
}


\author{
Kyu Ha Lee, Sounak Chakraborty, (Tony) Jianguo Sun \cr
Maintainer: Kyu Ha Lee <klee@hsph.harvard.edu>
}


\references{
Lee, K. H., Chakraborty, S., and Sun, J. (2011). 
Bayesian Variable Selection in Semiparametric Proportional Hazards Model for High Dimensional Survival Data. 
\emph{The International Journal of Biostatistics}, Volume 7, Issue 1, Pages 1-32. \cr

Lee, K. H., Chakraborty, S., and Sun, J. (2015). 
Survival Prediction and Variable Selection with Simultaneous Shrinkage and Grouping Priors. \emph{Statistical Analysis and Data Mining}, Volume 8, Issue 2, pages 114-127.\cr

Lee, K. H., Chakraborty, S., and Sun, J. (2017). 
Variable Selection for High-Dimensional Genomic Data with Censored Outcomes Using Group Lasso Prior. \emph{Computational Statistics and Data Analysis}, Volume 112, pages 1-13.\cr
}

%~~ Optionally other standard keywords, one per line, from file ~~
%~~ KEYWORDS in the R documentation directory ~~
\keyword{ package }
%\seealso{
%objects to See Also as \code{\link{help}}, ~~~
%}
%\examples{
%~~ simple examples of the most important functions ~~
%}
