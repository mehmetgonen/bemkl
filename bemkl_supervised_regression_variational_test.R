bemkl_supervised_regression_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]

  G <- list(mu = matrix(0, P, N), sigma = matrix(0, P, N))
  for (m in 1:P) {
    G$mu[m,] <- crossprod(state$a$mu, Km[,,m])
    G$sigma[m,] <- 1 / (state$upsilon$alpha * state$upsilon$beta) + diag(crossprod(Km[,,m], state$a$sigma) %*% Km[,,m])
  }
  
  y <- list(mu = matrix(0, N, 1), sigma = matrix(0, N, 1))
  y$mu <- crossprod(rbind(matrix(1, 1, N), G$mu), state$be$mu)
  y$sigma <- 1 / (state$epsilon$alpha * state$epsilon$beta) + diag(crossprod(rbind(matrix(1, 1, N), G$mu), state$be$sigma) %*% rbind(matrix(1, 1, N), G$mu))

  prediction <- list(G = G, y = y)
}
