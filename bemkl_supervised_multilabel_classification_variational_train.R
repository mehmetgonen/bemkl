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
  sigma_g <- parameters$sigma_g

  log2pi <- log(2 * pi)

  Lambda <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D, L), beta = matrix(parameters$beta_lambda, D, L))
  A <- list(mu = matrix(rnorm(D * L), D, L), sigma = array(diag(1, D, D), c(D, D, L)))
  G <- list(mu = (abs(array(rnorm(P * N * L), c(P, N, L))) + parameters$margin), sigma = diag(1, P, P))
  for (m in 1:P) {
    G$mu[m,,] <- G$mu[m,,] * sign(t(Y))
  }
  gamma <- list(alpha = matrix(parameters$alpha_gamma + 0.5, L, 1), beta = matrix(parameters$beta_gamma, L, 1))
  omega <- list(alpha = matrix(parameters$alpha_omega + 0.5, P, 1), beta = matrix(parameters$beta_omega, P, 1))
  be <- list(mu = rbind(matrix(0, L, 1), matrix(1, P, 1)), sigma = diag(1, L + P, L + P))
  F <- list(mu = (abs(matrix(rnorm(L * N), L, N)) + parameters$margin) * sign(Y), sigma = matrix(1, L, N))

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

  atimesaT.mu <- array(0, c(D, D, L))
  for (o in 1:L) {
    atimesaT.mu[,,o] <- tcrossprod(A$mu[,o], A$mu[,o]) + A$sigma[,,o]
  }
  GtimesGT.mu <- array(0, c(P, P, L))
  for (o in 1:L) {
    GtimesGT.mu[,,o] <- tcrossprod(G$mu[,,o], G$mu[,,o]) + N * G$sigma
  }
  btimesbT.mu <- tcrossprod(be$mu[1:L], be$mu[1:L]) + be$sigma[1:L, 1:L]
  etimeseT.mu <- tcrossprod(be$mu[(L + 1):(L + P)], be$mu[(L + 1):(L + P)]) + be$sigma[(L + 1):(L + P), (L + 1):(L + P)]
  etimesb.mu <- matrix(0, P, L)
  for (o in 1:L) {
    etimesb.mu[,o] <- be$mu[(L + 1):(L + P)] * be$mu[o] + be$sigma[(L + 1):(L + P), o]
  }
  KmtimesGT.mu <- matrix(0, D, L)
  for (o in 1:L) {
    KmtimesGT.mu[,o] <- Km %*% matrix(t(G$mu[,,o]), N * P, 1)
  }
  for (iter in 1:parameters$iteration) {
    # update Lambda
    for (o in 1:L) {
      Lambda$beta[,o] <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mu[,,o]))
    }
    # update A
    for (o in 1:L) {
      A$sigma[,,o] <- chol2inv(chol(diag(as.vector(Lambda$alpha[,o] * Lambda$beta[,o]), D, D) + KmKm / sigma_g^2))
      A$mu[,o] <- A$sigma[,,o] %*% KmtimesGT.mu[,o] / sigma_g^2
      atimesaT.mu[,,o] <- tcrossprod(A$mu[,o], A$mu[,o]) + A$sigma[,,o]
    }
    # update G
    G$sigma <- chol2inv(chol(diag(1, P, P) / sigma_g^2 + etimeseT.mu))
    for (o in 1:L) {
      G$mu[,,o] <- G$sigma %*% (t(matrix(crossprod(A$mu[,o], Km), N, P)) / sigma_g^2 + tcrossprod(be$mu[(L + 1):(L + P)], F$mu[o,]) - matrix(etimesb.mu[,o], P, N, byrow = FALSE))
      GtimesGT.mu[,,o] <- tcrossprod(G$mu[,,o], G$mu[,,o]) + N * G$sigma
      KmtimesGT.mu[,o] <- Km %*% matrix(t(G$mu[,,o]), N * P, 1)
    }
    # update gamma
    gamma$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * diag(btimesbT.mu))
    # update omega
    omega$beta <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mu))
    # update b and e
    be$sigma <- rbind(cbind(diag(as.vector(gamma$alpha * gamma$beta), L, L) + N * diag(1, L, L), t(apply(G$mu, c(1, 3), sum))), cbind(apply(G$mu, c(1, 3), sum), diag(as.vector(omega$alpha * omega$beta), P, P)))
    for (o in 1:L) {
      be$sigma[(L + 1):(L + P), (L + 1):(L + P)] <- be$sigma[(L + 1):(L + P), (L + 1):(L + P)] + GtimesGT.mu[,,o]
    }
    be$sigma <- chol2inv(chol(be$sigma))
    be$mu <- matrix(0, L + P, 1)
    be$mu[1:L] <- rowSums(F$mu)
    for (o in 1:L) {
      be$mu[(L + 1):(L + P)] <- be$mu[(L + 1):(L + P)] + G$mu[,,o] %*% F$mu[o,]
    }
    be$mu <- be$sigma %*% be$mu
    btimesbT.mu <- tcrossprod(be$mu[1:L], be$mu[1:L]) + be$sigma[1:L, 1:L]
    etimeseT.mu <- tcrossprod(be$mu[(L + 1):(L + P)], be$mu[(L + 1):(L + P)]) + be$sigma[(L + 1):(L + P), (L + 1):(L + P)]
    for (o in 1:L) {
        etimesb.mu[,o] <- be$mu[(L + 1):(L + P)] * be$mu[o] + be$sigma[(L + 1):(L + P), o]
    }
    # update F
    output <- matrix(0, L, N)
    for (o in 1:L) {
      output[o,] <- crossprod(rbind(matrix(1, 1, N), G$mu[,,o]), be$mu[c(o, (L + 1):(L + P))])
    }
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    F$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    F$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2

    if (parameters$progress == 1) {
      lb <- 0

      # p(Lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(Lambda$alpha) + log(Lambda$beta)) - Lambda$alpha * Lambda$beta / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(A | Lambda)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(as.vector(Lambda$alpha[,o] * Lambda$beta[,o]) * diag(atimesaT.mu[,,o])) - 0.5 * (D * log2pi - sum(digamma(Lambda$alpha[,o]) + log(Lambda$beta[,o])))
      }
      # p(G | A, Km)
      for (o in 1:L) {
        lb <- lb - 0.5 * sigma_g^-2 * sum(diag(GtimesGT.mu[,,o])) + sigma_g^-2 * crossprod(A$mu[,o], KmtimesGT.mu[,o]) - 0.5 * sigma_g^-2 * sum(KmKm * atimesaT.mu[,,o]) - 0.5 * N * P * (log2pi + 2 * log(sigma_g))
      }
      # p(gamma)
      lb <- lb + sum((parameters$alpha_gamma - 1) * (digamma(gamma$alpha) + log(gamma$beta)) - gamma$alpha * gamma$beta / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma))
      # p(b | gamma)
      lb <- lb - 0.5 * sum(as.vector(gamma$alpha * gamma$beta) * diag(btimesbT.mu)) - 0.5 * (L * log2pi - sum(digamma(gamma$alpha) + log(gamma$beta)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$alpha) + log(omega$beta)) - omega$alpha * omega$beta / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$alpha * omega$beta) * diag(etimeseT.mu)) - 0.5 * (P * log2pi - sum(digamma(omega$alpha) + log(omega$beta)))
      # p(F | b, e, G)
      for (o in 1:L) {
        lb <- lb - 0.5 * (crossprod(F$mu[o,], F$mu[o,]) + sum(F$sigma[o,])) + crossprod(F$mu[o,], crossprod(G$mu[,,o], be$mu[(L + 1):(L + P)])) + sum(be$mu[o] * F$mu[o,]) - 0.5 * sum(etimeseT.mu * GtimesGT.mu[,,o]) - sum(crossprod(G$mu[,,o], etimesb.mu[,o])) - 0.5 * N * btimesbT.mu[o,o] - 0.5 * N * log2pi
      }

      # q(Lambda)
      lb <- lb + sum(Lambda$alpha + log(Lambda$beta) + lgamma(Lambda$alpha) + (1 - Lambda$alpha) * digamma(Lambda$alpha))
      # q(A)
      for (o in 1:L) {
        lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(A$sigma[,,o]))
      }
      # q(G)
      lb <- lb + 0.5 * L * (N * (P * (log2pi + 1) + logdet(G$sigma)))
      # q(gamma)
      lb <- lb + sum(gamma$alpha + log(gamma$beta) + lgamma(gamma$alpha) + (1 - gamma$alpha) * digamma(gamma$alpha))
      # q(omega)
      lb <- lb + sum(omega$alpha + log(omega$beta) + lgamma(omega$alpha) + (1 - omega$alpha) * digamma(omega$alpha))
      # q(b, e)
      lb <- lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be$sigma))
      # q(F)
      lb <- lb + 0.5 * sum(log2pi + F$sigma) + sum(log(normalization))

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