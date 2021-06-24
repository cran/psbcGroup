\name{VS}
\alias{VS}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
 Function to perform variable selection using SNC-BIC thresholding method
}
\description{
The \code{VS} is a function to perform variable selection using SNC-BIC thresholding method
}
\usage{
VS(fit, X, psiVec=seq(0.001, 1, 0.001))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{fit}{
    an object of class \code{aftGL}, \code{psbcEN}, \code{psbcFL}, or \code{psbcGL}.
}
  \item{X}{
    a covariate matrix, \code{n} observations by \code{p} variables\cr
}
  \item{psiVec}{
    a vector of candidate threshold values for the SNC step
}
}

\references{
Lee, K. H., Chakraborty, S., and Sun, J. (2017). 
Variable Selection for High-Dimensional Genomic Data with Censored Outcomes Using Group Lasso Prior. \emph{Computational Statistics and Data Analysis}, Volume 112, pages 1-13.\cr
}

\author{
Kyu Ha Lee
}


\seealso{
\code{\link{psbcEN}}, \code{\link{psbcFL}}, \code{\link{psbcGL}}, \code{\link{aftGL}}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
