# Mehmet Gonen (mehmet.gonen@gmail.com)

bemkl_supervised_multilabel_classification_variational_test <- function(Km, state) {
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  L <- length(state$be$mean) - P

  G <- list(mean = array(0, c(P, N, L)), covariance = array(0, c(P, N, L)))
  for (o in 1:L) {
    for (m in 1:P) {
      G$mean[m,,o] <- crossprod(state$A$mean[,o], Km[,,m])
      G$covariance[m,,o] <- state$parameters$sigmag^2 + diag(crossprod(Km[,,m], state$A$covariance[,,o]) %*% Km[,,m])
    }
  }
  
  F <- list(mean = matrix(0, L, N), covariance = matrix(0, L, N))
  for (o in 1:L) {
    F$mean[o,] <- crossprod(state$be$mean[c(o, (L + 1):(L + P))], rbind(matrix(1, 1, N), G$mean[,,o]))
    F$covariance[o,] <- 1 + diag(crossprod(rbind(matrix(1, 1, N), G$mean[,,o]), state$be$covariance[c(o, (L + 1):(L + P)), c(o, (L + 1):(L + P))]) %*% rbind(matrix(1, 1, N), G$mean[,,o]))
  }

  pos <- 1 - pnorm((+state$parameters$margin - F$mean) / F$covariance)
  neg <- pnorm((-state$parameters$margin - F$mean) / F$covariance)
  P <- pos / (pos + neg)

  prediction <- list(F = F, P = P)
}