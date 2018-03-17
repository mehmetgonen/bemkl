bemkl_supervised_multioutput_regression_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  L <- length(state$be$mu) - P

  G <- list(mu = array(0, c(P, N, L)), sigma = array(0, c(P, N, L)))
  for (o in 1:L) {
    for (m in 1:P) {
      G$mu[m,,o] <- crossprod(state$A$mu[,o], Km[,,m])
      G$sigma[m,,o] <- 1 / (state$upsilon$alpha[o] * state$upsilon$beta[o]) + diag(crossprod(Km[,,m], state$A$sigma[,,o]) %*% Km[,,m])
    }
  }
  
  Y <- list(mu = matrix(0, L, N), sigma = matrix(0, L, N))
  for (o in 1:L) {
    Y$mu[o,] <- crossprod(state$be$mu[c(o, (L + 1):(L + P))], rbind(matrix(1, 1, N), G$mu[,,o]))
    Y$sigma[o,] <- 1 / (state$epsilon$alpha[o] * state$epsilon$beta[o]) + diag(crossprod(rbind(matrix(1, 1, N), G$mu[,,o]), state$be$sigma[c(o, (L + 1):(L + P)), c(o, (L + 1):(L + P))]) %*% rbind(matrix(1, 1, N), G$mu[,,o]))
  }

  prediction <- list(G = G, Y = Y)
}
