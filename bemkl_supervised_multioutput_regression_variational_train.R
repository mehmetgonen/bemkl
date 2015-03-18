# Mehmet Gonen (mehmet.gonen@gmail.com)

logdet <- function(Sigma) {
    2 * sum(log(diag(chol(Sigma))))
}

bemkl_supervised_multioutput_regression_variational_train <- function(Km, Y, parameters) {
  set.seed(parameters$seed)

  D <- dim(Km)[1]
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  L <- dim(Y)[1]

  log2pi <- log(2 * pi)

  Lambda <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D, L), beta = matrix(parameters$beta_lambda, D, L))
  upsilon <- list(alpha = matrix(parameters$alpha_upsilon + 0.5 * N * P, L, 1), beta = matrix(parameters$beta_upsilon, L, 1))
  A <- list(mu = matrix(rnorm(D * L), D, L), sigma = array(diag(1, D, D), c(D, D, L)))
  G <- list(mu = array(rnorm(P * N * L), c(P, N, L)), sigma = array(diag(1, P, P), c(P, P, L)))
  gamma <- list(alpha = matrix(parameters$alpha_gamma + 0.5, L, 1), beta = matrix(parameters$beta_gamma, L, 1))
  omega <- list(alpha = matrix(parameters$alpha_omega + 0.5, P, 1), beta = matrix(parameters$beta_omega, P, 1))
  epsilon <- list(alpha = matrix(parameters$alpha_epsilon + 0.5 * N, L, 1), beta = matrix(parameters$beta_epsilon, L, 1))
  be <- list(mu = rbind(matrix(0, L, 1), matrix(1, P, 1)), sigma = diag(1, L + P, L + P))

  KmKm <- matrix(0, D, D)
  for(m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)
  
  if (parameters$progress == 1) {
    bounds <- matrix(0, parameters$iteration, 1)
  }

  atimesaT.mu <- array(0, c(D, D, L))
  for (o in 1:L) {
    atimesaT.mu[,,o] <- tcrossprod(A$mu[,o], A$mu[,o]) + A$sigma[,,o]
  }
  GtimesGT.mu <- array(0, c(P, P, L))
  for (o in 1:L) {
    GtimesGT.mu[,,o] <- tcrossprod(G$mu[,,o], G$mu[,,o]) + N * G$sigma[,,o]
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
    # update upsilon
    for (o in 1:L) {
      upsilon$beta[o] <- 1 / (1 / parameters$beta_upsilon + 0.5 * (sum(diag(GtimesGT.mu[,,o])) - 2 * sum(matrix(crossprod(A$mu[,o], Km), N, P) * t(G$mu[,,o])) + sum(KmKm * atimesaT.mu[,,o])))
    }
    # update A
    for (o in 1:L) {
      A$sigma[,,o] <- chol2inv(chol(diag(as.vector(Lambda$alpha[,o] * Lambda$beta[,o]), D, D) + upsilon$alpha[o] * upsilon$beta[o] * KmKm))
      A$mu[,o] <- A$sigma[,,o] %*% (upsilon$alpha[o] * upsilon$beta[o] * KmtimesGT.mu[,o])
      atimesaT.mu[,,o] <- tcrossprod(A$mu[,o], A$mu[,o]) + A$sigma[,,o]
    }
    # update G
    for (o in 1:L) {
      G$sigma[,,o] <- chol2inv(chol(diag(upsilon$alpha[o] * upsilon$beta[o], P, P) + epsilon$alpha[o] * epsilon$beta[o] * etimeseT.mu))
      G$mu[,,o] <- G$sigma[,,o] %*% (upsilon$alpha[o] * upsilon$beta[o] * t(matrix(crossprod(A$mu[,o], Km), N, P)) + epsilon$alpha[o] * epsilon$beta[o] * (tcrossprod(be$mu[(L + 1):(L + P)], Y[o,]) - matrix(etimesb.mu[,o], P, N, byrow = FALSE)))
      GtimesGT.mu[,,o] <- tcrossprod(G$mu[,,o], G$mu[,,o]) + N * G$sigma[,,o]
      KmtimesGT.mu[,o] <- Km %*% matrix(t(G$mu[,,o]), N * P, 1)
    }
    # update gamma
    gamma$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * diag(btimesbT.mu))
    # update omega
    omega$beta <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mu))
    # update epsilon
    for (o in 1:L) {
      epsilon$beta[o] <- 1 / (1 / parameters$beta_epsilon + 0.5 * as.double(crossprod(Y[o,], Y[o,]) - 2 * tcrossprod(Y[o,], rbind(matrix(1, 1, N), G$mu[,,o])) %*% be$mu[c(o, (L + 1):(L + P))] + N * btimesbT.mu[o, o] + sum(GtimesGT.mu[,,o] * etimeseT.mu) + 2 * crossprod(rowSums(G$mu[,,o]), etimesb.mu[,o])))
    }
    # update b and e
    be$sigma <- rbind(cbind(diag(as.vector(gamma$alpha * gamma$beta), L, L) + N * diag(as.vector(epsilon$alpha * epsilon$beta), L, L), matrix(epsilon$alpha * epsilon$beta, L, P, byrow = FALSE) * t(apply(G$mu, c(1, 3), sum))), cbind(matrix(epsilon$alpha * epsilon$beta, P, L, byrow = TRUE) * apply(G$mu, c(1, 3), sum), diag(as.vector(omega$alpha * omega$beta), P, P)))
    for (o in 1:L) {
      be$sigma[(L + 1):(L + P), (L + 1):(L + P)] <- be$sigma[(L + 1):(L + P), (L + 1):(L + P)] + epsilon$alpha[o] * epsilon$beta[o] * GtimesGT.mu[,,o]
    }
    be$sigma <- chol2inv(chol(be$sigma))
    be$mu <- matrix(0, L + P, 1)
    be$mu[1:L] <- epsilon$alpha * epsilon$beta * rowSums(Y)
    for (o in 1:L) {
      be$mu[(L + 1):(L + P)] <- be$mu[(L + 1):(L + P)] + epsilon$alpha[o] * epsilon$beta[o] * G$mu[,,o] %*% Y[o,]
    }
    be$mu <- be$sigma %*% be$mu
    btimesbT.mu <- tcrossprod(be$mu[1:L], be$mu[1:L]) + be$sigma[1:L, 1:L]
    etimeseT.mu <- tcrossprod(be$mu[(L + 1):(L + P)], be$mu[(L + 1):(L + P)]) + be$sigma[(L + 1):(L + P), (L + 1):(L + P)]
    for (o in 1:L) {
      etimesb.mu[,o] <- be$mu[(L + 1):(L + P)] * be$mu[o] + be$sigma[(L + 1):(L + P), o]
    }

    if (parameters$progress == 1) {
      lb <- 0

      # p(Lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(Lambda$alpha) + log(Lambda$beta)) - Lambda$alpha * Lambda$beta / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(upsilon)
      lb <- lb + sum((parameters$alpha_upsilon - 1) * (digamma(upsilon$alpha) + log(upsilon$beta)) - upsilon$alpha * upsilon$beta / parameters$beta_upsilon - lgamma(parameters$alpha_upsilon) - parameters$alpha_upsilon * log(parameters$beta_upsilon))
      # p(A | Lambda)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(as.vector(Lambda$alpha[,o] * Lambda$beta[,o]) * diag(atimesaT.mu[,,o])) - 0.5 * (D * log2pi - sum(log(Lambda$alpha[,o] * Lambda$beta[,o])))
      }
      # p(G | A, Km, upsilon)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(diag(GtimesGT.mu[,,o])) * upsilon$alpha[o] * upsilon$beta[o] + crossprod(A$mu[,o], KmtimesGT.mu[,o]) * upsilon$alpha[o] * upsilon$beta[o] - 0.5 * sum(KmKm * atimesaT.mu[,,o]) * upsilon$alpha[o] * upsilon$beta[o] - 0.5 * N * P * (log2pi - log(upsilon$alpha[o] * upsilon$beta[o]))
      }
      # p(gamma)
      lb <- lb + sum((parameters$alpha_gamma - 1) * (digamma(gamma$alpha) + log(gamma$beta)) - gamma$alpha * gamma$beta / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma))
      # p(b | gamma)
      lb <- lb - 0.5 * sum(as.vector(gamma$alpha * gamma$beta) * diag(btimesbT.mu)) - 0.5 * (L * log2pi - sum(log(gamma$alpha * gamma$beta)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$alpha) + log(omega$beta)) - omega$alpha * omega$beta / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$alpha * omega$beta) * diag(etimeseT.mu)) - 0.5 * (P * log2pi - sum(log(omega$alpha * omega$beta)))
      # p(epsilon)
      lb <- lb + sum((parameters$alpha_epsilon - 1) * (digamma(epsilon$alpha) + log(epsilon$beta)) - epsilon$alpha * epsilon$beta / parameters$beta_epsilon - lgamma(parameters$alpha_epsilon) - parameters$alpha_epsilon * log(parameters$beta_epsilon))
      # p(Y | b, e, G, epsilon)
      for (o in 1:L) {
        lb <- lb - 0.5 * crossprod(Y[o,], Y[o,]) * epsilon$alpha[o] * epsilon$beta[o] + crossprod(Y[o,], crossprod(G$mu[,,o], be$mu[(L + 1):(L + P)])) * epsilon$alpha[o] * epsilon$beta[o] + sum(be$mu[o] * Y[o,]) * epsilon$alpha[o] * epsilon$beta[o] - 0.5 * sum(etimeseT.mu * GtimesGT.mu[,,o]) * epsilon$alpha[o] * epsilon$beta[o] - sum(crossprod(G$mu[,,o], etimesb.mu[,o])) * epsilon$alpha[o] * epsilon$beta[o] - 0.5 * N * btimesbT.mu[o, o] * epsilon$alpha[o] * epsilon$beta[o] - 0.5 * N * (log2pi - log(epsilon$alpha[o] * epsilon$beta[o]))
      }

      # q(Lambda)
      lb <- lb + sum(Lambda$alpha + log(Lambda$beta) + lgamma(Lambda$alpha) + (1 - Lambda$alpha) * digamma(Lambda$alpha))
      # q(upsilon)
      lb <- lb + sum(upsilon$alpha + log(upsilon$beta) + lgamma(upsilon$alpha) + (1 - upsilon$alpha) * digamma(upsilon$alpha))
      # q(A)
      for (o in 1:L) {
        lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(A$sigma[,,o]))
      }
      # q(G)
      for (o in 1:L) {
        lb <- lb + 0.5 * N * (P * (log2pi + 1) + logdet(G$sigma[,,o]))
      }
      # q(gamma)
      lb <- lb + sum(gamma$alpha + log(gamma$beta) + lgamma(gamma$alpha) + (1 - gamma$alpha) * digamma(gamma$alpha))
      # q(omega)
      lb <- lb + sum(omega$alpha + log(omega$beta) + lgamma(omega$alpha) + (1 - omega$alpha) * digamma(omega$alpha))
      # q(epsilon)
      lb <- lb + sum(epsilon$alpha + log(epsilon$beta) + lgamma(epsilon$alpha) + (1 - epsilon$alpha) * digamma(epsilon$alpha))
      # q(b, e)
      lb <- lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be$sigma))

      bounds[iter] <- lb
    }
  }
  
  if (parameters$progress == 1) {
    state <- list(Lambda = Lambda, upsilon = upsilon, A = A, gamma = gamma, omega = omega, epsilon = epsilon, be = be, bounds = bounds, parameters = parameters)
  }
  else {
    state <- list(Lambda = Lambda, upsilon = upsilon, A = A, gamma = gamma, omega = omega, epsilon = epsilon, be = be, parameters = parameters)
  }
}