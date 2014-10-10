# Mehmet Gonen (mehmet.gonen@gmail.com)

bemkl_supervised_classification_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]

  G <- list(mean = matrix(0, P, N), covariance = matrix(0, P, N))
  for (m in 1:P) {
    G$mean[m,] <- crossprod(state$a$mean, Km[,,m])
    G$covariance[m,] <- state$parameters$sigmag^2 + diag(crossprod(Km[,,m], state$a$covariance) %*% Km[,,m])
  }
  
  f <- list(mean = matrix(0, N, 1), covariance = matrix(0, N, 1))
  f$mean <- crossprod(rbind(matrix(1, 1, N), G$mean), state$be$mean)
  f$covariance <- 1 + diag(crossprod(rbind(matrix(1, 1, N), G$mean), state$be$covariance) %*% rbind(matrix(1, 1, N), G$mean))

  pos <- 1 - pnorm((+state$parameters$margin - f$mean) / f$covariance)
  neg <- pnorm((-state$parameters$margin - f$mean) / f$covariance)
  p <- pos / (pos + neg)

  prediction <- list(G = G, f = f, p = p)
}