bemkl_supervised_classification_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]

  G <- list(mu = matrix(0, P, N), sigma = matrix(0, P, N))
  for (m in 1:P) {
    G$mu[m,] <- crossprod(state$a$mu, Km[,,m])
    G$sigma[m,] <- state$parameters$sigma_g^2 + diag(crossprod(Km[,,m], state$a$sigma) %*% Km[,,m])
  }
  
  f <- list(mu = matrix(0, N, 1), sigma = matrix(0, N, 1))
  f$mu <- crossprod(rbind(matrix(1, 1, N), G$mu), state$be$mu)
  f$sigma <- 1 + diag(crossprod(rbind(matrix(1, 1, N), G$mu), state$be$sigma) %*% rbind(matrix(1, 1, N), G$mu))

  pos <- 1 - pnorm((+state$parameters$margin - f$mu) / f$sigma)
  neg <- pnorm((-state$parameters$margin - f$mu) / f$sigma)
  p <- pos / (pos + neg)

  prediction <- list(G = G, f = f, p = p)
}
