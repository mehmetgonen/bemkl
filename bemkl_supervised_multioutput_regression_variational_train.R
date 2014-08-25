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

  Lambda <- list(shape = matrix(parameters$alpha_lambda + 0.5, D, L), scale = matrix(parameters$beta_lambda, D, L))
  upsilon <- list(shape = matrix(parameters$alpha_upsilon + 0.5 * N * P, L, 1), scale = matrix(parameters$beta_upsilon, L, 1))
  A <- list(mean = matrix(rnorm(D * L), D, L), covariance = array(diag(1, D, D), c(D, D, L)))
  G <- list(mean = array(rnorm(P * N * L), c(P, N, L)), covariance = array(diag(1, P, P), c(P, P, L)))
  gamma <- list(shape = matrix(parameters$alpha_gamma + 0.5, L, 1), scale = matrix(parameters$beta_gamma, L, 1))
  omega <- list(shape = matrix(parameters$alpha_omega + 0.5, P, 1), scale = matrix(parameters$beta_omega, P, 1))
  epsilon <- list(shape = matrix(parameters$alpha_epsilon + 0.5 * N, L, 1), scale = matrix(parameters$beta_epsilon, L, 1))
  be <- list(mean = rbind(matrix(0, L, 1), matrix(1, P, 1)), covariance = diag(1, L + P, L + P))

  KmKm <- matrix(0, D, D)
  for(m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)
  
  if (parameters$progress == 1) {
    bounds <- matrix(0, parameters$iteration, 1)
  }

  atimesaT.mean <- array(0, c(D, D, L))
  for (o in 1:L) {
    atimesaT.mean[,,o] <- tcrossprod(A$mean[,o], A$mean[,o]) + A$covariance[,,o]
  }
  GtimesGT.mean <- array(0, c(P, P, L))
  for (o in 1:L) {
    GtimesGT.mean[,,o] <- tcrossprod(G$mean[,,o], G$mean[,,o]) + N * G$covariance[,,o]
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
    # update upsilon
    for (o in 1:L) {
      upsilon$scale[o] <- 1 / (1 / parameters$beta_upsilon + 0.5 * (sum(diag(GtimesGT.mean[,,o])) - 2 * sum(matrix(crossprod(A$mean[,o], Km), N, P) * t(G$mean[,,o])) + sum(KmKm * atimesaT.mean[,,o])))
    }
    # update A
    for (o in 1:L) {
      A$covariance[,,o] <- chol2inv(chol(diag(as.vector(Lambda$shape[,o] * Lambda$scale[,o]), D, D) + upsilon$shape[o] * upsilon$scale[o] * KmKm))
      A$mean[,o] <- A$covariance[,,o] %*% (upsilon$shape[o] * upsilon$scale[o] * KmtimesGT.mean[,o])
      atimesaT.mean[,,o] <- tcrossprod(A$mean[,o], A$mean[,o]) + A$covariance[,,o]
    }
    # update G
    for (o in 1:L) {
      G$covariance[,,o] <- chol2inv(chol(diag(upsilon$shape[o] * upsilon$scale[o], P, P) + epsilon$shape[o] * epsilon$scale[o] * etimeseT.mean))
      G$mean[,,o] <- G$covariance[,,o] %*% (upsilon$shape[o] * upsilon$scale[o] * t(matrix(crossprod(A$mean[,o], Km), N, P)) + epsilon$shape[o] * epsilon$scale[o] * (tcrossprod(be$mean[(L + 1):(L + P)], Y[o,]) - matrix(etimesb.mean[,o], P, N, byrow = FALSE)))
      GtimesGT.mean[,,o] <- tcrossprod(G$mean[,,o], G$mean[,,o]) + N * G$covariance[,,o]
      KmtimesGT.mean[,o] <- Km %*% matrix(t(G$mean[,,o]), N * P, 1)
    }
    # update gamma
    gamma$scale <- 1 / (1 / parameters$beta_gamma + 0.5 * diag(btimesbT.mean))
    # update omega
    omega$scale <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mean))
    # update epsilon
    for (o in 1:L) {
      epsilon$scale[o] <- 1 / (1 / parameters$beta_epsilon + 0.5 * as.double(crossprod(Y[o,], Y[o,]) - 2 * tcrossprod(Y[o,], rbind(matrix(1, 1, N), G$mean[,,o])) %*% be$mean[c(o, (L + 1):(L + P))] + N * btimesbT.mean[o, o] + sum(GtimesGT.mean[,,o] * etimeseT.mean) + 2 * crossprod(rowSums(G$mean[,,o]), etimesb.mean[,o])))
    }
    # update b and e
    be$covariance <- rbind(cbind(diag(as.vector(gamma$shape * gamma$scale), L, L) + N * diag(as.vector(epsilon$shape * epsilon$scale), L, L), matrix(epsilon$shape * epsilon$scale, L, P, byrow = FALSE) * t(apply(G$mean, c(1, 3), sum))), cbind(matrix(epsilon$shape * epsilon$scale, P, L, byrow = TRUE) * apply(G$mean, c(1, 3), sum), diag(as.vector(omega$shape * omega$scale), P, P)))
    for (o in 1:L) {
      be$covariance[(L + 1):(L + P), (L + 1):(L + P)] <- be$covariance[(L + 1):(L + P), (L + 1):(L + P)] + epsilon$shape[o] * epsilon$scale[o] * GtimesGT.mean[,,o]
    }
    be$covariance <- chol2inv(chol(be$covariance))
    be$mean <- matrix(0, L + P, 1)
    be$mean[1:L] <- epsilon$shape * epsilon$scale * rowSums(Y)
    for (o in 1:L) {
      be$mean[(L + 1):(L + P)] <- be$mean[(L + 1):(L + P)] + epsilon$shape[o] * epsilon$scale[o] * G$mean[,,o] %*% Y[o,]
    }
    be$mean <- be$covariance %*% be$mean
    btimesbT.mean <- tcrossprod(be$mean[1:L], be$mean[1:L]) + be$covariance[1:L, 1:L]
    etimeseT.mean <- tcrossprod(be$mean[(L + 1):(L + P)], be$mean[(L + 1):(L + P)]) + be$covariance[(L + 1):(L + P), (L + 1):(L + P)]
    for (o in 1:L) {
      etimesb.mean[,o] <- be$mean[(L + 1):(L + P)] * be$mean[o] + be$covariance[(L + 1):(L + P), o]
    }

    if (parameters$progress == 1) {
      lb <- 0

      # p(Lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(Lambda$shape) + log(Lambda$scale)) - Lambda$shape * Lambda$scale / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(upsilon)
      lb <- lb + sum((parameters$alpha_upsilon - 1) * (digamma(upsilon$shape) + log(upsilon$scale)) - upsilon$shape * upsilon$scale / parameters$beta_upsilon - lgamma(parameters$alpha_upsilon) - parameters$alpha_upsilon * log(parameters$beta_upsilon))
      # p(A | Lambda)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(as.vector(Lambda$shape[,o] * Lambda$scale[,o]) * diag(atimesaT.mean[,,o])) - 0.5 * (D * log2pi - sum(log(Lambda$shape[,o] * Lambda$scale[,o])))
      }
      # p(G | A, Km, upsilon)
      for (o in 1:L) {
        lb <- lb - 0.5 * sum(diag(GtimesGT.mean[,,o])) * upsilon$shape[o] * upsilon$scale[o] + crossprod(A$mean[,o], KmtimesGT.mean[,o]) * upsilon$shape[o] * upsilon$scale[o] - 0.5 * sum(KmKm * atimesaT.mean[,,o]) * upsilon$shape[o] * upsilon$scale[o] - 0.5 * N * P * (log2pi - log(upsilon$shape[o] * upsilon$scale[o]))
      }
      # p(gamma)
      lb <- lb + sum((parameters$alpha_gamma - 1) * (digamma(gamma$shape) + log(gamma$scale)) - gamma$shape * gamma$scale / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma))
      # p(b | gamma)
      lb <- lb - 0.5 * sum(as.vector(gamma$shape * gamma$scale) * diag(btimesbT.mean)) - 0.5 * (L * log2pi - sum(log(gamma$shape * gamma$scale)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$shape) + log(omega$scale)) - omega$shape * omega$scale / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$shape * omega$scale) * diag(etimeseT.mean)) - 0.5 * (P * log2pi - sum(log(omega$shape * omega$scale)))
      # p(epsilon)
      lb <- lb + sum((parameters$alpha_epsilon - 1) * (digamma(epsilon$shape) + log(epsilon$scale)) - epsilon$shape * epsilon$scale / parameters$beta_epsilon - lgamma(parameters$alpha_epsilon) - parameters$alpha_epsilon * log(parameters$beta_epsilon))
      # p(Y | b, e, G, epsilon)
      for (o in 1:L) {
        lb <- lb - 0.5 * crossprod(Y[o,], Y[o,]) * epsilon$shape[o] * epsilon$scale[o] + crossprod(Y[o,], crossprod(G$mean[,,o], be$mean[(L + 1):(L + P)])) * epsilon$shape[o] * epsilon$scale[o] + sum(be$mean[o] * Y[o,]) * epsilon$shape[o] * epsilon$scale[o] - 0.5 * sum(etimeseT.mean * GtimesGT.mean[,,o]) * epsilon$shape[o] * epsilon$scale[o] - sum(crossprod(G$mean[,,o], etimesb.mean[,o])) * epsilon$shape[o] * epsilon$scale[o] - 0.5 * N * btimesbT.mean[o, o] * epsilon$shape[o] * epsilon$scale[o] - 0.5 * N * (log2pi - log(epsilon$shape[o] * epsilon$scale[o]))
      }

      # q(Lambda)
      lb <- lb + sum(Lambda$shape + log(Lambda$scale) + lgamma(Lambda$shape) + (1 - Lambda$shape) * digamma(Lambda$shape))
      # q(upsilon)
      lb <- lb + sum(upsilon$shape + log(upsilon$scale) + lgamma(upsilon$shape) + (1 - upsilon$shape) * digamma(upsilon$shape))
      # q(A)
      for (o in 1:L) {
        lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(A$covariance[,,o]))
      }
      # q(G)
      for (o in 1:L) {
        lb <- lb + 0.5 * N * (P * (log2pi + 1) + logdet(G$covariance[,,o]))
      }
      # q(gamma)
      lb <- lb + sum(gamma$shape + log(gamma$scale) + lgamma(gamma$shape) + (1 - gamma$shape) * digamma(gamma$shape))
      # q(omega)
      lb <- lb + sum(omega$shape + log(omega$scale) + lgamma(omega$shape) + (1 - omega$shape) * digamma(omega$shape))
      # q(epsilon)
      lb <- lb + sum(epsilon$shape + log(epsilon$scale) + lgamma(epsilon$shape) + (1 - epsilon$shape) * digamma(epsilon$shape))
      # q(b, e)
      lb <- lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be$covariance))

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