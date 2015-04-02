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

  lambda <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D, 1), beta = matrix(parameters$beta_lambda, D, 1))
  upsilon <- list(alpha = parameters$alpha_upsilon + 0.5 * N * P, beta = parameters$beta_upsilon)
  a <- list(mu = matrix(rnorm(D), D, 1), sigma = diag(1, D, D))
  G <- list(mu = matrix(rnorm(P * N), P, N), sigma = diag(1, P, P))
  gamma <- list(alpha = parameters$alpha_gamma + 0.5, beta = parameters$beta_gamma)
  omega <- list(alpha = matrix(parameters$alpha_omega + 0.5, P, 1), beta = matrix(parameters$beta_omega, P, 1))
  epsilon <- list(alpha = parameters$alpha_epsilon + 0.5 * N, beta = parameters$beta_epsilon)
  be <- list(mu = rbind(0, matrix(1, P, 1)), sigma = diag(1, P + 1, P + 1))

  KmKm <- matrix(0, D, D)
  for(m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)

  if (parameters$progress == 1) {
    bounds <- matrix(0, parameters$iteration, 1)
  }

  atimesaT.mu <- tcrossprod(a$mu, a$mu) + a$sigma
  GtimesGT.mu <- tcrossprod(G$mu, G$mu) + N * G$sigma
  btimesbT.mu <- be$mu[1]^2 + be$sigma[1, 1]
  etimeseT.mu <- tcrossprod(be$mu[2:(P + 1)], be$mu[2:(P + 1)]) + be$sigma[2:(P + 1), 2:(P + 1)]
  etimesb.mu <- be$mu[2:(P + 1)] * be$mu[1] + be$sigma[2:(P + 1), 1]
  KmtimesGT.mu <- Km %*% matrix(t(G$mu), N * P, 1)
  for (iter in 1:parameters$iteration) {
    # update lambda
    lambda$beta <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mu))
    # update upsilon
    upsilon$beta <- 1 / (1 / parameters$beta_upsilon + 0.5 * (sum(diag(GtimesGT.mu)) - 2 * sum(matrix(crossprod(a$mu, Km), N, P) * t(G$mu)) + sum(KmKm * atimesaT.mu)))
    # update a
    a$sigma <- chol2inv(chol(diag(as.vector(lambda$alpha * lambda$beta), D, D) + upsilon$alpha * upsilon$beta * KmKm))
    a$mu <- a$sigma %*% (upsilon$alpha * upsilon$beta * KmtimesGT.mu)
    atimesaT.mu <- tcrossprod(a$mu, a$mu) + a$sigma
    # update G
    G$sigma <- chol2inv(chol(diag(upsilon$alpha * upsilon$beta, P, P) + epsilon$alpha * epsilon$beta * etimeseT.mu))
    G$mu <- G$sigma %*% (upsilon$alpha * upsilon$beta * t(matrix(crossprod(a$mu, Km), N, P)) + epsilon$alpha * epsilon$beta * (tcrossprod(be$mu[2:(P + 1)], y) - matrix(etimesb.mu, P, N, byrow = FALSE)))
    GtimesGT.mu <- tcrossprod(G$mu, G$mu) + N * G$sigma
    KmtimesGT.mu <- Km %*% matrix(t(G$mu), N * P, 1)
    # update gamma
    gamma$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * btimesbT.mu)
    # update omega
    omega$beta <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mu))
    # update epsilon
    epsilon$beta <- 1 / (1 / parameters$beta_epsilon + 0.5 * as.double(crossprod(y, y) - 2 * crossprod(y, crossprod(rbind(matrix(1, 1, N), G$mu), be$mu)) + N * btimesbT.mu + sum(GtimesGT.mu * etimeseT.mu) + 2 * crossprod(rowSums(G$mu), etimesb.mu)))
    # update b and e
    be$sigma <- chol2inv(chol(rbind(cbind(gamma$alpha * gamma$beta + epsilon$alpha * epsilon$beta * N, epsilon$alpha * epsilon$beta * t(rowSums(G$mu))), cbind(epsilon$alpha * epsilon$beta * rowSums(G$mu), diag(as.vector(omega$alpha * omega$beta), P, P) + epsilon$alpha * epsilon$beta * GtimesGT.mu))))
    be$mu <- be$sigma %*% (epsilon$alpha * epsilon$beta * rbind(matrix(1, 1, N), G$mu) %*% y)
    btimesbT.mu <- be$mu[1]^2 + be$sigma[1, 1]
    etimeseT.mu <- tcrossprod(be$mu[2:(P + 1)], be$mu[2:(P + 1)]) + be$sigma[2:(P + 1), 2:(P + 1)]
    etimesb.mu <- be$mu[2:(P + 1)] * be$mu[1] + be$sigma[2:(P + 1), 1]
    
    if (parameters$progress == 1) {
      lb <- 0
      
      # p(lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(lambda$alpha) + log(lambda$beta)) - lambda$alpha * lambda$beta / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(upsilon)
      lb <- lb + (parameters$alpha_upsilon - 1) * (digamma(upsilon$alpha) + log(upsilon$beta)) - upsilon$alpha * upsilon$beta / parameters$beta_upsilon - lgamma(parameters$alpha_upsilon) - parameters$alpha_upsilon * log(parameters$beta_upsilon)
      # p(a | lambda)
      lb <- lb - 0.5 * sum(as.vector(lambda$alpha * lambda$beta) * diag(atimesaT.mu)) - 0.5 * (D * log2pi - sum(digamma(lambda$alpha) + log(lambda$beta)))
      # p(G | a, Km, upsilon)
      lb <- lb - 0.5 * sum(diag(GtimesGT.mu)) * upsilon$alpha * upsilon$beta + crossprod(a$mu, KmtimesGT.mu) * upsilon$alpha * upsilon$beta - 0.5 * sum(KmKm * atimesaT.mu) * upsilon$alpha * upsilon$beta - 0.5 * N * P * (log2pi - (digamma(upsilon$alpha) + log(upsilon$beta)))
      # p(gamma)
      lb <- lb + (parameters$alpha_gamma - 1) * (digamma(gamma$alpha) + log(gamma$beta)) - gamma$alpha * gamma$beta / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma)
      # p(b | gamma)
      lb <- lb - 0.5 * gamma$alpha * gamma$beta * btimesbT.mu - 0.5 * (log2pi - (digamma(gamma$alpha) + log(gamma$beta)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$alpha) + log(omega$beta)) - omega$alpha * omega$beta / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$alpha * omega$beta) * diag(etimeseT.mu)) - 0.5 * (P * log2pi - sum(digamma(omega$alpha) + log(omega$beta)))
      # p(epsilon)
      lb <- lb + (parameters$alpha_epsilon - 1) * (digamma(epsilon$alpha) + log(epsilon$beta)) - epsilon$alpha * epsilon$beta / parameters$beta_epsilon - lgamma(parameters$alpha_epsilon) - parameters$alpha_epsilon * log(parameters$beta_epsilon)
      # p(y | b, e, G, epsilon)
      lb <- lb - 0.5 * crossprod(y, y) * epsilon$alpha * epsilon$beta + crossprod(y, crossprod(G$mu, be$mu[2:(P + 1)])) * epsilon$alpha * epsilon$beta + sum(be$mu[1] * y) * epsilon$alpha * epsilon$beta - 0.5 * sum(etimeseT.mu * GtimesGT.mu) * epsilon$alpha * epsilon$beta - sum(crossprod(G$mu, etimesb.mu)) * epsilon$alpha * epsilon$beta - 0.5 * N * btimesbT.mu * epsilon$alpha * epsilon$beta - 0.5 * N * (log2pi - (digamma(epsilon$alpha) + log(epsilon$beta)))

      # q(lambda)
      lb <- lb + sum(lambda$alpha + log(lambda$beta) + lgamma(lambda$alpha) + (1 - lambda$alpha) * digamma(lambda$alpha))
      # q(upsilon)
      lb <- lb + upsilon$alpha + log(upsilon$beta) + lgamma(upsilon$alpha) + (1 - upsilon$alpha) * digamma(upsilon$alpha)
      # q(a)
      lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(a$sigma))
      # q(G)
      lb <- lb + 0.5 * N * (P * (log2pi + 1) + logdet(G$sigma))
      # q(gamma)
      lb <- lb + gamma$alpha + log(gamma$beta) + lgamma(gamma$alpha) + (1 - gamma$alpha) * digamma(gamma$alpha)
      # q(omega)
      lb <- lb + sum(omega$alpha + log(omega$beta) + lgamma(omega$alpha) + (1 - omega$alpha) * digamma(omega$alpha))
      # q(epsilon)
      lb <- lb + epsilon$alpha + log(epsilon$beta) + lgamma(epsilon$alpha) + (1 - epsilon$alpha) * digamma(epsilon$alpha)
      # q(b, e)
      lb <- lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be$sigma))
      
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