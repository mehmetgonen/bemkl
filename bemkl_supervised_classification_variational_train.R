# Mehmet Gonen (mehmet.gonen@gmail.com)

logdet <- function(Sigma) {
    2 * sum(log(diag(chol(Sigma))))
}

bemkl_supervised_classification_variational_train <- function(Km, y, parameters) {    
  set.seed(parameters$seed)

  D <- dim(Km)[1]
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  sigma_g <- parameters$sigma_g

  log2pi <- log(2 * pi)

  lambda <- list(alpha = matrix(parameters$alpha_lambda + 0.5, D, 1), beta = matrix(parameters$beta_lambda, D, 1))
  a <- list(mu = matrix(rnorm(D), D, 1), sigma = diag(1, D, D))
  G <- list(mu = (abs(matrix(rnorm(P * N), P, N)) + parameters$margin) * sign(matrix(y, P, N, byrow = TRUE)), sigma = diag(1, P, P))
  gamma <- list(alpha = parameters$alpha_gamma + 0.5, beta = parameters$beta_gamma)
  omega <- list(alpha = matrix(parameters$alpha_omega + 0.5, P, 1), beta = matrix(parameters$beta_omega, P, 1))
  be <- list(mu = rbind(0, matrix(1, P, 1)), sigma = diag(1, P + 1, P + 1))
  f <- list(mu = (abs(matrix(rnorm(N), N, 1)) + parameters$margin) * sign(y), sigma = matrix(1, N, 1))

  KmKm <- matrix(0, D, D)
  for (m in 1:P) {
    KmKm <- KmKm + tcrossprod(Km[,,m], Km[,,m])
  }
  Km <- matrix(Km, D, N * P)

  lower <- matrix(-1e40, N, 1)
  lower[which(y > 0)] <- +parameters$margin
  upper <- matrix(+1e40, N, 1)
  upper[which(y < 0)] <- -parameters$margin

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
    # update a
    a$sigma <- chol2inv(chol(diag(as.vector(lambda$alpha * lambda$beta), D, D) + KmKm / sigma_g^2))
    a$mu <- a$sigma %*% KmtimesGT.mu / sigma_g^2
    atimesaT.mu <- tcrossprod(a$mu, a$mu) + a$sigma
    # update G
    G$sigma <- chol2inv(chol(diag(1, P, P) / sigma_g^2 + etimeseT.mu))
    G$mu <- G$sigma %*% (t(matrix(crossprod(a$mu, Km), N, P)) / sigma_g^2 + tcrossprod(be$mu[2:(P + 1)], f$mu) - matrix(etimesb.mu, P, N, byrow = FALSE))
    GtimesGT.mu <- tcrossprod(G$mu, G$mu) + N * G$sigma
    KmtimesGT.mu <- Km %*% matrix(t(G$mu), N * P, 1)
    # update gamma
    gamma$beta <- 1 / (1 / parameters$beta_gamma + 0.5 * btimesbT.mu)
    # update omega
    omega$beta <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mu))
    # update b and e
    be$sigma <- chol2inv(chol(rbind(cbind(gamma$alpha * gamma$beta + N, t(rowSums(G$mu))), cbind(rowSums(G$mu), diag(as.vector(omega$alpha * omega$beta), P, P) + GtimesGT.mu))))
    be$mu <- be$sigma %*% (rbind(matrix(1, 1, N), G$mu) %*% f$mu)
    btimesbT.mu <- be$mu[1]^2 + be$sigma[1, 1]
    etimeseT.mu <- tcrossprod(be$mu[2:(P + 1)], be$mu[2:(P + 1)]) + be$sigma[2:(P + 1), 2:(P + 1)]
    etimesb.mu <- be$mu[2:(P + 1)] * be$mu[1] + be$sigma[2:(P + 1), 1]
    # update f
    output <- crossprod(rbind(matrix(1, 1, N), G$mu), be$mu)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    f$mu <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    f$sigma <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2

    if (parameters$progress == 1) {
      lb <- 0

      # p(lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(lambda$alpha) + log(lambda$beta)) - lambda$alpha * lambda$beta / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(a | lambda)
      lb <- lb - 0.5 * sum(as.vector(lambda$alpha * lambda$beta) * diag(atimesaT.mu)) - 0.5 * (D * log2pi - sum(digamma(lambda$alpha) + log(lambda$beta)))
      # p(G | a, Km)
      lb <- lb - 0.5 * sigma_g^-2 * sum(diag(GtimesGT.mu)) + sigma_g^-2 * crossprod(a$mu, KmtimesGT.mu) - 0.5 * sigma_g^-2 * sum(KmKm * atimesaT.mu) - 0.5 * N * P * (log2pi + 2 * log(sigma_g))
      # p(gamma)
      lb <- lb + (parameters$alpha_gamma - 1) * (digamma(gamma$alpha) + log(gamma$beta)) - gamma$alpha * gamma$beta / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma)
      # p(b | gamma)
      lb <- lb - 0.5 * gamma$alpha * gamma$beta * btimesbT.mu - 0.5 * (log2pi - (digamma(gamma$alpha) + log(gamma$beta)))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$alpha) + log(omega$beta)) - omega$alpha * omega$beta / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$alpha * omega$beta) * diag(etimeseT.mu)) - 0.5 * (P * log2pi - sum(digamma(omega$alpha) + log(omega$beta)))
      # p(f | b, e, G)
      lb <- lb - 0.5 * (crossprod(f$mu, f$mu) + sum(f$sigma)) + crossprod(f$mu, crossprod(G$mu, be$mu[2:(P + 1)])) + sum(be$mu[1] * f$mu) - 0.5 * sum(etimeseT.mu * GtimesGT.mu) - sum(crossprod(G$mu, etimesb.mu)) - 0.5 * N * btimesbT.mu - 0.5 * N * log2pi

      # q(lambda)
      lb <- lb + sum(lambda$alpha + log(lambda$beta) + lgamma(lambda$alpha) + (1 - lambda$alpha) * digamma(lambda$alpha))
      # q(a)
      lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(a$sigma))
      # q(G)
      lb <- lb + 0.5 * N * (P * (log2pi + 1) + logdet(G$sigma))
      # q(gamma)
      lb <- lb + gamma$alpha + log(gamma$beta) + lgamma(gamma$alpha) + (1 - gamma$alpha) * digamma(gamma$alpha)
      # q(omega)
      lb <- lb + sum(omega$alpha + log(omega$beta) + lgamma(omega$alpha) + (1 - omega$alpha) * digamma(omega$alpha))
      # q(b, e)
      lb <- lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be$sigma))
      # q(f)
      lb <- lb + 0.5 * sum(log2pi + f$sigma) + sum(log(normalization))
      
      bounds[iter] <- lb
    }
  }
  
  if (parameters$progress == 1) {
    state <- list(lambda = lambda, a = a, gamma = gamma, omega = omega, be = be, bounds = bounds, parameters = parameters)
  }
  else {
    state <- list(lambda = lambda, a = a, gamma = gamma, omega = omega, be = be, parameters = parameters)
  }
}