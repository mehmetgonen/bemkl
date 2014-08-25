# Mehmet Gonen (mehmet.gonen@gmail.com)

logdet <- function(Sigma) {
    2 * sum(log(diag(chol(Sigma))))
}

bemkl_supervised_regression_variational_train <- function(Km, y, parameters) {    
  set.seed(parameters$seed)

  D <- dim(Km)[1]
  N <- dim(Km)[2]
  P <- dim(Km)[3]

  log2pi <- log(2 * pi)

  lambda <- list(shape = matrix(parameters$alpha_lambda + 0.5, D, 1), scale = matrix(parameters$beta_lambda, D, 1))
  upsilon <- list(shape = parameters$alpha_upsilon + 0.5 * N * P, scale = parameters$beta_upsilon)
  a <- list(mean = matrix(rnorm(D), D, 1), covariance = diag(1, D, D))
  G <- list(mean = matrix(rnorm(P * N), P, N), covariance = diag(1, P, P))
  gamma <- list(shape = parameters$alpha_gamma + 0.5, scale = parameters$beta_gamma)
  omega <- list(shape = matrix(parameters$alpha_omega + 0.5, P, 1), scale = matrix(parameters$beta_omega, P, 1))
  epsilon <- list(shape = parameters$alpha_epsilon + 0.5 * N, scale = parameters$beta_epsilon)
  be <- list(mean = rbind(0, matrix(1, P, 1)), covariance = diag(1, P + 1, P + 1))

  KmKm <- matrix(0, D, D)
  for(m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)

  if (parameters$progress == 1) {
    bounds <- matrix(0, parameters$iteration, 1)
  }

  atimesaT.mean <- tcrossprod(a$mean, a$mean) + a$covariance
  GtimesGT.mean <- tcrossprod(G$mean, G$mean) + N * G$covariance
  btimesbT.mean <- be$mean[1]^2 + be$covariance[1, 1]
  etimeseT.mean <- tcrossprod(be$mean[2:(P + 1)], be$mean[2:(P + 1)]) + be$covariance[2:(P + 1), 2:(P + 1)]
  etimesb.mean <- be$mean[2:(P + 1)] * be$mean[1] + be$covariance[2:(P + 1), 1]
  KmtimesGT.mean <- Km %*% matrix(t(G$mean), N * P, 1)
  for (iter in 1:parameters$iteration) {
    # update lambda
    lambda$scale <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mean))
    # update upsilon
    upsilon$scale <- 1 / (1 / parameters$beta_upsilon + 0.5 * (sum(diag(GtimesGT.mean)) - 2 * sum(matrix(crossprod(a$mean, Km), N, P) * t(G$mean)) + sum(KmKm * atimesaT.mean)))
    # update a
    a$covariance <- chol2inv(chol(diag(as.vector(lambda$shape * lambda$scale), D, D) + upsilon$shape * upsilon$scale * KmKm))
    a$mean <- a$covariance %*% (upsilon$shape * upsilon$scale * KmtimesGT.mean)
    atimesaT.mean <- tcrossprod(a$mean, a$mean) + a$covariance
    # update G
    G$covariance <- chol2inv(chol(diag(upsilon$shape * upsilon$scale, P, P) + epsilon$shape * epsilon$scale * etimeseT.mean))
    G$mean <- G$covariance %*% (upsilon$shape * upsilon$scale * t(matrix(crossprod(a$mean, Km), N, P)) + epsilon$shape * epsilon$scale * (tcrossprod(be$mean[2:(P + 1)], y) - matrix(etimesb.mean, P, N, byrow = FALSE)))
    GtimesGT.mean <- tcrossprod(G$mean, G$mean) + N * G$covariance
    KmtimesGT.mean <- Km %*% matrix(t(G$mean), N * P, 1)
    # update gamma
    gamma$scale <- 1 / (1 / parameters$beta_gamma + 0.5 * btimesbT.mean)
    # update omega
    omega$scale <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mean))
    # update epsilon
    epsilon$scale <- 1 / (1 / parameters$beta_epsilon + 0.5 * as.double(crossprod(y, y) - 2 * crossprod(y, crossprod(rbind(matrix(1, 1, N), G$mean), be$mean)) + N * btimesbT.mean + sum(GtimesGT.mean * etimeseT.mean) + 2 * crossprod(rowSums(G$mean), etimesb.mean)))
    # update b and e
    be$covariance <- chol2inv(chol(rbind(cbind(gamma$shape * gamma$scale + epsilon$shape * epsilon$scale * N, epsilon$shape * epsilon$scale * t(rowSums(G$mean))), cbind(epsilon$shape * epsilon$scale * rowSums(G$mean), diag(as.vector(omega$shape * omega$scale), P, P) + epsilon$shape * epsilon$scale * GtimesGT.mean))))
    be$mean <- be$covariance %*% (epsilon$shape * epsilon$scale * rbind(matrix(1, 1, N), G$mean) %*% y)
    btimesbT.mean <- be$mean[1]^2 + be$covariance[1, 1]
    etimeseT.mean <- tcrossprod(be$mean[2:(P + 1)], be$mean[2:(P + 1)]) + be$covariance[2:(P + 1), 2:(P + 1)]
    etimesb.mean <- be$mean[2:(P + 1)] * be$mean[1] + be$covariance[2:(P + 1), 1]
    
    if (parameters$progress == 1) {
      lb <- 0
      
      # p(lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(lambda$shape) + log(lambda$scale)) - lambda$shape * lambda$scale / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(upsilon)
      lb <- lb + (parameters$alpha_upsilon - 1) * (digamma(upsilon$shape) + log(upsilon$scale)) - upsilon$shape * upsilon$scale / parameters$beta_upsilon - lgamma(parameters$alpha_upsilon) - parameters$alpha_upsilon * log(parameters$beta_upsilon)
      # p(a | lambda)
      lb <- lb - 0.5 * sum(as.vector(lambda$shape * lambda$scale) * diag(atimesaT.mean)) - 0.5 * (D * log2pi - sum(log(lambda$shape * lambda$scale)))
      # p(G | a, Km, upsilon)
      lb <- lb - 0.5 * sum(diag(GtimesGT.mean)) * upsilon$shape * upsilon$scale + crossprod(a$mean, KmtimesGT.mean) * upsilon$shape * upsilon$scale - 0.5 * sum(KmKm * atimesaT.mean) * upsilon$shape * upsilon$scale - 0.5 * N * P * (log2pi - log(upsilon$shape * upsilon$scale))
      # p(gamma)
      lb <- lb + (parameters$alpha_gamma - 1) * (digamma(gamma$shape) + log(gamma$scale)) - gamma$shape * gamma$scale / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma)
      # p(b | gamma)
      lb <- lb - 0.5 * gamma$shape * gamma$scale * btimesbT.mean - 0.5 * (log2pi - log(gamma$shape * gamma$scale))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$shape) + log(omega$scale)) - omega$shape * omega$scale / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$shape * omega$scale) * diag(etimeseT.mean)) - 0.5 * (P * log2pi - sum(log(omega$shape * omega$scale)))
      # p(epsilon)
      lb <- lb + (parameters$alpha_epsilon - 1) * (digamma(epsilon$shape) + log(epsilon$scale)) - epsilon$shape * epsilon$scale / parameters$beta_epsilon - lgamma(parameters$alpha_epsilon) - parameters$alpha_epsilon * log(parameters$beta_epsilon)
      # p(y | b, e, G, epsilon)
      lb <- lb - 0.5 * crossprod(y, y) * epsilon$shape * epsilon$scale + crossprod(y, crossprod(G$mean, be$mean[2:(P + 1)])) * epsilon$shape * epsilon$scale + sum(be$mean[1] * y) * epsilon$shape * epsilon$scale - 0.5 * sum(etimeseT.mean * GtimesGT.mean) * epsilon$shape * epsilon$scale - sum(crossprod(G$mean, etimesb.mean)) * epsilon$shape * epsilon$scale - 0.5 * N * btimesbT.mean * epsilon$shape * epsilon$scale - 0.5 * N * (log2pi - log(epsilon$shape * epsilon$scale))

      # q(lambda)
      lb <- lb + sum(lambda$shape + log(lambda$scale) + lgamma(lambda$shape) + (1 - lambda$shape) * digamma(lambda$shape))
      # q(upsilon)
      lb <- lb + upsilon$shape + log(upsilon$scale) + lgamma(upsilon$shape) + (1 - upsilon$shape) * digamma(upsilon$shape)
      # q(a)
      lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(a$covariance))
      # q(G)
      lb <- lb + 0.5 * N * (P * (log2pi + 1) + logdet(G$covariance))
      # q(gamma)
      lb <- lb + gamma$shape + log(gamma$scale) + lgamma(gamma$shape) + (1 - gamma$shape) * digamma(gamma$shape)
      # q(omega)
      lb <- lb + sum(omega$shape + log(omega$scale) + lgamma(omega$shape) + (1 - omega$shape) * digamma(omega$shape))
      # q(epsilon)
      lb <- lb + epsilon$shape + log(epsilon$scale) + lgamma(epsilon$shape) + (1 - epsilon$shape) * digamma(epsilon$shape)
      # q(b, e)
      lb <- lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be$covariance))
      
      bounds[iter] <- lb
    }
  }
  
  if (parameters$progress == 1) {
    state <- list(lambda = lambda, upsilon = upsilon, a = a, gamma = gamma, omega = omega, epsilon = epsilon, be = be, bounds = bounds, parameters = parameters)
  }
  else {
    state <- list(lambda = lambda, upsilon = upsilon, a = a, gamma = gamma, omega = omega, epsilon = epsilon, be = be, parameters = parameters)
  }
}