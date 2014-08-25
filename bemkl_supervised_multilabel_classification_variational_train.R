# Mehmet Gonen (mehmet.gonen@gmail.com)

logdet <- function(Sigma) {
    2 * sum(log(diag(chol(Sigma))))
}

bemkl_supervised_multilabel_classification_variational_train <- function(Km, Y, parameters) {
  set.seed(parameters$seed)

  D <- dim(Km)[1]
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  L <- dim(Y)[1]
  sigmag <- parameters$sigmag

  log2pi <- log(2 * pi)

  Lambda <- list(shape = matrix(parameters$alpha_lambda + 0.5, D, L), scale = matrix(parameters$beta_lambda, D, L))
  A <- list(mean = matrix(rnorm(D * L), D, L), covariance = array(diag(1, D, D), c(D, D, L)))
  G <- list(mean = (abs(array(rnorm(P * N * L), c(P, N, L))) + parameters$margin), covariance = diag(1, P, P))
  for (m in 1:P) {
    G$mean[m,,] <- G$mean[m,,] * sign(t(Y))
  }
  gamma <- list(shape = matrix(parameters$alpha_gamma + 0.5, L, 1), scale = matrix(parameters$beta_gamma, L, 1))
  omega <- list(shape = matrix(parameters$alpha_omega + 0.5, P, 1), scale = matrix(parameters$beta_omega, P, 1))
  be <- list(mean = rbind(matrix(0, L, 1), matrix(1, P, 1)), covariance = diag(1, L + P, L + P))
  F <- list(mean = (abs(matrix(rnorm(L * N), L, N)) + parameters$margin) * sign(Y), covariance = matrix(1, L, N))

  KmKm <- matrix(0, D, D)
  for (m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)

  lower <- matrix(-1e40, L, N)
  lower[which(Y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, L, N)
  upper[which(Y < 0)] <- -parameters$margin
  
  if (parameters$progress == 1) {
    bounds <- matrix(0, parameters$iteration, 1)
  }

  atimesaT.mean <- array(0, c(D, D, L))
  for (o in 1:L) {
    atimesaT.mean[,,o] <- tcrossprod(A$mean[,o], A$mean[,o]) + A$covariance[,,o]
  }
  GtimesGT.mean <- array(0, c(P, P, L))
  for (o in 1:L) {
    GtimesGT.mean[,,o] <- tcrossprod(G$mean[,,o], G$mean[,,o]) + N * G$covariance
  }
  btimesbT.mean <- tcrossprod(be$mean[1:L], be$mean[1:L]) + be$covariance[1:L, 1:L]
  etimeseT.mean <- tcrossprod(be$mean[(L + 1):(L + P)], be$mean[(L + 1):(L + P)]) + be$covariance[(L + 1):(L + P), (L + 1):(L + P)]
  etimesb.mean <- matrix(0, P, L)
  for (o in 1:L) {
    etimesb.mean[,o] <- be$mean[(L + 1):(L + P)] * be$mean[o] + be$covariance[(L + 1):(L + P), o]
  }
  KmtimesGT.mean <- matrix(0, D, L)
  for (o in 1:L) {
    KmtimesGT.mean[,o] <- Km %*% matrix(t(G$mean[,,o]), N * P, 1)
  }
  for (iter in 1:parameters$iteration) {
    # update Lambda
    for (o in 1:L) {
      Lambda$scale[,o] <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mean[,,o]))
    }
    # update A
    for (o in 1:L) {
      A$covariance[,,o] <- chol2inv(chol(diag(as.vector(Lambda$shape[,o] * Lambda$scale[,o]), D, D) + KmKm / sigmag^2))
      A$mean[,o] <- A$covariance[,,o] %*% KmtimesGT.mean[,o] / sigmag^2
      atimesaT.mean[,,o] <- tcrossprod(A$mean[,o], A$mean[,o]) + A$covariance[,,o]
    }
    # update G
    G$covariance <- chol2inv(chol(diag(1, P, P) / sigmag^2 + etimeseT.mean))
    for (o in 1:L) {
      G$mean[,,o] <- G$covariance %*% (t(matrix(crossprod(A$mean[,o], Km), N, P)) / sigmag^2 + tcrossprod(be$mean[(L + 1):(L + P)], F$mean[o,]) - matrix(etimesb.mean[,o], P, N, byrow = FALSE))
      GtimesGT.mean[,,o] <- tcrossprod(G$mean[,,o], G$mean[,,o]) + N * G$covariance
      KmtimesGT.mean[,o] <- Km %*% matrix(t(G$mean[,,o]), N * P, 1)
    }
    # update gamma
    gamma$scale <- 1 / (1 / parameters$beta_gamma + 0.5 * diag(btimesbT.mean))
    # update omega
    omega$scale <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mean))
    # update b and e
    be$covariance <- rbind(cbind(diag(as.vector(gamma$shape * gamma$scale), L, L) + N * diag(1, L, L), t(apply(G$mean, c(1, 3), sum))), cbind(apply(G$mean, c(1, 3), sum), diag(as.vector(omega$shape * omega$scale), P, P)))
    for (o in 1:L) {
      be$covariance[(L + 1):(L + P), (L + 1):(L + P)] <- be$covariance[(L + 1):(L + P), (L + 1):(L + P)] + GtimesGT.mean[,,o]
    }
    be$covariance <- chol2inv(chol(be$covariance))
    be$mean <- matrix(0, L + P, 1)
    be$mean[1:L] <- rowSums(F$mean)
    for (o in 1:L) {
      be$mean[(L + 1):(L + P)] <- be$mean[(L + 1):(L + P)] + G$mean[,,o] %*% F$mean[o,]
    }
    be$mean <- be$covariance %*% be$mean
    btimesbT.mean <- tcrossprod(be$mean[1:L], be$mean[1:L]) + be$covariance[1:L, 1:L]
    etimeseT.mean <- tcrossprod(be$mean[(L + 1):(L + P)], be$mean[(L + 1):(L + P)]) + be$covariance[(L + 1):(L + P), (L + 1):(L + P)]
    for (o in 1:L) {
        etimesb.mean[,o] <- be$mean[(L + 1):(L + P)] * be$mean[o] + be$covariance[(L + 1):(L + P), o]
    }
    # update F
    output <- matrix(0, L, N)
    for (o in 1:L) {
      output[o,] <- crossprod(rbind(matrix(1, 1, N), G$mean[,,o]), be$mean[c(o, (L + 1):(L + P))])
    }
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mean <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$covariance <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2

    if (parameters$progress == 1) {
      lb <- 0

      # p(Lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(Lambda$shape) + log(Lambda$scale)) - Lambda$shape * Lambda$scale / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(A | Lambda)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(as.vector(Lambda$shape[,o] * Lambda$scale[,o]) * diag(atimesaT.mean[,,o])) - 0.5 * (D * log2pi - sum(log(Lambda$shape[,o] * Lambda$scale[,o])))
      }
      # p(G | A, Km)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(diag(GtimesGT.mean[,,o])) + crossprod(A$mean[,o], KmtimesGT.mean[,o]) - 0.5 * sum(KmKm * atimesaT.mean[,,o]) - 0.5 * N * P * (log2pi + 2 * log(sigmag))
      }
      # p(gamma)
      lb <- lb + sum((parameters$alpha_gamma - 1) * (digamma(gamma$shape) + log(gamma$scale)) - gamma$shape * gamma$scale / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma))
      # p(b | gamma)
      lb <- lb - 0.5 * sum(as.vector(gamma$shape * gamma$scale) * diag(btimesbT.mean)) - 0.5 * (L * log2pi - sum(log(gamma$shape * gamma$scale)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$shape) + log(omega$scale)) - omega$shape * omega$scale / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$shape * omega$scale) * diag(etimeseT.mean)) - 0.5 * (P * log2pi - sum(log(omega$shape * omega$scale)))
      # p(F | b, e, G)
      for (o in 1:L) {
        lb <- lb - 0.5 * (crossprod(F$mean[o,], F$mean[o,]) + sum(F$covariance[o,])) + crossprod(F$mean[o,], crossprod(G$mean[,,o], be$mean[(L + 1):(L + P)])) + sum(be$mean[o] * F$mean[o,]) - 0.5 * sum(etimeseT.mean * GtimesGT.mean[,,o]) - sum(crossprod(G$mean[,,o], etimesb.mean[,o])) - 0.5 * N * btimesbT.mean[o,o] - 0.5 * N * log2pi
      }

      # q(Lambda)
      lb <- lb + sum(Lambda$shape + log(Lambda$scale) + lgamma(Lambda$shape) + (1 - Lambda$shape) * digamma(Lambda$shape))
      # q(A)
      for (o in 1:L) {
        lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(A$covariance[,,o]))
      }
      # q(G)
      lb <- lb + 0.5 * L * (N * (P * (log2pi + 1) + logdet(G$covariance)))
      # q(gamma)
      lb <- lb + sum(gamma$shape + log(gamma$scale) + lgamma(gamma$shape) + (1 - gamma$shape) * digamma(gamma$shape))
      # q(omega)
      lb <- lb + sum(omega$shape + log(omega$scale) + lgamma(omega$shape) + (1 - omega$shape) * digamma(omega$shape))
      # q(b, e)
      lb <- lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be$covariance))
      # q(F)
      lb <- lb + 0.5 * sum(log2pi + F$covariance) + sum(log(normalization))

      bounds[iter] <- lb
    }
  }
  
  if (parameters$progress == 1) {
    state <- list(Lambda = Lambda, A = A, gamma = gamma, omega = omega, be = be, bounds = bounds, parameters = parameters)
  }
  else {
    state <- list(Lambda = Lambda, A = A, gamma = gamma, omega = omega, be = be, parameters = parameters)
  }
}