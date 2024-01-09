\name{survData}
\alias{survData}
\docType{data}
\title{
A simulated survival dataset.
}
\description{
Univariate survival data.
}
\usage{data(survData)}
\format{
  a data frame with 2000 observations on the following 4 variables.
  \describe{
    \item{\code{time}}{the time to event}
    \item{\code{event}}{the censoring indicators for the event time; 1=event observed, 0=censored}
    \item{\code{cluster}}{cluster numbers}    
    \item{\code{cov1}}{the first column of covariate matrix x}
    \item{\code{cov2}}{the second column of covariate matrix x}
  }
}
%\details{
%%  ~~ If necessary, more details than the __description__ above ~~
%}
%\source{
%%  ~~ reference to a publication or URL from which the data were obtained ~~
%}
%\references{
%%  ~~ possibly secondary sources and usages ~~
%}
\examples{
data(survData)
}
\keyword{datasets}

