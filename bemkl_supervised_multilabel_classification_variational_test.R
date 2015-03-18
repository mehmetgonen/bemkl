# Mehmet Gonen (mehmet.gonen@gmail.com)

bemkl_supervised_multilabel_classification_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  L <- length(state$be$mu) - P

  G <- list(mu = array(0, c(P, N, L)), sigma = array(0, c(P, N, L)))
  for (o in 1:L) {
    for (m in 1:P) {
      G$mu[m,,o] <- crossprod(state$A$mu[,o], Km[,,m])
      G$sigma[m,,o] <- state$parameters$sigma_g^2 + diag(crossprod(Km[,,m], state$A$sigma[,,o]) %*% Km[,,m])
    }
  }
  
  F <- list(mu = matrix(0, L, N), sigma = matrix(0, L, N))
  for (o in 1:L) {
    F$mu[o,] <- crossprod(state$be$mu[c(o, (L + 1):(L + P))], rbind(matrix(1, 1, N), G$mu[,,o]))
    F$sigma[o,] <- 1 + diag(crossprod(rbind(matrix(1, 1, N), G$mu[,,o]), state$be$sigma[c(o, (L + 1):(L + P)), c(o, (L + 1):(L + P))]) %*% rbind(matrix(1, 1, N), G$mu[,,o]))
  }

  pos <- 1 - pnorm((+state$parameters$margin - F$mu) / F$sigma)
  neg <- pnorm((-state$parameters$margin - F$mu) / F$sigma)
  P <- pos / (pos + neg)

  prediction <- list(G = G, F = F, P = P)
}