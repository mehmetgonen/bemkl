# Mehmet Gonen (mehmet.gonen@gmail.com)

bemkl_supervised_regression_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]

  G <- list(mean = matrix(0, P, N), covariance = matrix(0, P, N))
  for (m in 1:P) {
    G$mean[m,] <- crossprod(state$a$mean, Km[,,m])
    G$covariance[m,] <- 1 / (state$upsilon$shape * state$upsilon$scale) + diag(crossprod(Km[,,m], state$a$covariance) %*% Km[,,m])
  }
  
  y <- list(mean = matrix(0, N, 1), covariance = matrix(0, N, 1))
  y$mean <- crossprod(rbind(matrix(1, 1, N), G$mean), state$be$mean)
  y$covariance <- 1 / (state$epsilon$shape * state$epsilon$scale) + diag(crossprod(rbind(matrix(1, 1, N), G$mean), state$be$covariance) %*% rbind(matrix(1, 1, N), G$mean))

  prediction <- list(G = G, y = y)
}