# Mehmet Gonen (mehmet.gonen@gmail.com)

logdet <- function(Sigma) {
    2 * sum(log(diag(chol(Sigma))))
}

bemkl_supervised_classification_variational_train <- function(Km, y, parameters) {    
  set.seed(parameters$seed)

  D <- dim(Km)[1]
  N <- dim(Km)[2]
  P <- dim(Km)[3]
  sigmag <- parameters$sigmag

  log2pi <- log(2 * pi)

  lambda <- list(shape = matrix(parameters$alpha_lambda + 0.5, D, 1), scale = matrix(parameters$beta_lambda, D, 1))
  a <- list(mean = matrix(rnorm(D), D, 1), covariance = diag(1, D, D))
  G <- list(mean = (abs(matrix(rnorm(P * N), P, N)) + parameters$margin) * sign(matrix(y, P, N, byrow = TRUE)), covariance = diag(1, P, P))
  gamma <- list(shape = parameters$alpha_gamma + 0.5, scale = parameters$beta_gamma)
  omega <- list(shape = matrix(parameters$alpha_omega + 0.5, P, 1), scale = matrix(parameters$beta_omega, P, 1))
  be <- list(mean = rbind(0, matrix(1, P, 1)), covariance = diag(1, P + 1, P + 1))
  f <- list(mean = (abs(matrix(rnorm(N), N, 1)) + parameters$margin) * sign(y), covariance = matrix(1, N, 1))

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

  atimesaT.mean <- tcrossprod(a$mean, a$mean) + a$covariance
  GtimesGT.mean <- tcrossprod(G$mean, G$mean) + N * G$covariance
  btimesbT.mean <- be$mean[1]^2 + be$covariance[1, 1]
  etimeseT.mean <- tcrossprod(be$mean[2:(P + 1)], be$mean[2:(P + 1)]) + be$covariance[2:(P + 1), 2:(P + 1)]
  etimesb.mean <- be$mean[2:(P + 1)] * be$mean[1] + be$covariance[2:(P + 1), 1]
  KmtimesGT.mean <- Km %*% matrix(t(G$mean), N * P, 1)
  for (iter in 1:parameters$iteration) {
    # update lambda
    lambda$scale <- 1 / (1 / parameters$beta_lambda + 0.5 * diag(atimesaT.mean))
    # update a
    a$covariance <- chol2inv(chol(diag(as.vector(lambda$shape * lambda$scale), D, D) + KmKm / sigmag^2))
    a$mean <- a$covariance %*% KmtimesGT.mean / sigmag^2
    atimesaT.mean <- tcrossprod(a$mean, a$mean) + a$covariance
    # update G
    G$covariance <- chol2inv(chol(diag(1, P, P) / sigmag^2 + etimeseT.mean))
    G$mean <- G$covariance %*% (t(matrix(crossprod(a$mean, Km), N, P)) / sigmag^2 + tcrossprod(be$mean[2:(P + 1)], f$mean) - matrix(etimesb.mean, P, N, byrow = FALSE))
    GtimesGT.mean <- tcrossprod(G$mean, G$mean) + N * G$covariance
    KmtimesGT.mean <- Km %*% matrix(t(G$mean), N * P, 1)
    # update gamma
    gamma$scale <- 1 / (1 / parameters$beta_gamma + 0.5 * btimesbT.mean)
    # update omega
    omega$scale <- 1 / (1 / parameters$beta_omega + 0.5 * diag(etimeseT.mean))
    # update b and e
    be$covariance <- chol2inv(chol(rbind(cbind(gamma$shape * gamma$scale + N, t(rowSums(G$mean))), cbind(rowSums(G$mean), diag(as.vector(omega$shape * omega$scale), P, P) + GtimesGT.mean))))
    be$mean <- be$covariance %*% (rbind(matrix(1, 1, N), G$mean) %*% f$mean)
    btimesbT.mean <- be$mean[1]^2 + be$covariance[1, 1]
    etimeseT.mean <- tcrossprod(be$mean[2:(P + 1)], be$mean[2:(P + 1)]) + be$covariance[2:(P + 1), 2:(P + 1)]
    etimesb.mean <- be$mean[2:(P + 1)] * be$mean[1] + be$covariance[2:(P + 1), 1]
    # update f
    output <- crossprod(rbind(matrix(1, 1, N), G$mean), be$mean)
    alpha_norm <- lower - output
    beta_norm <- upper - output
    normalization <- pnorm(beta_norm) - pnorm(alpha_norm)
    normalization[which(normalization == 0)] <- 1
    f$mean <- output + (dnorm(alpha_norm) - dnorm(beta_norm)) / normalization
    f$covariance <- 1 + (alpha_norm * dnorm(alpha_norm) - beta_norm * dnorm(beta_norm)) / normalization - (dnorm(alpha_norm) - dnorm(beta_norm))^2 / normalization^2

    if (parameters$progress == 1) {
      lb <- 0

      # p(lambda)
      lb <- lb + sum((parameters$alpha_lambda - 1) * (digamma(lambda$shape) + log(lambda$scale)) - lambda$shape * lambda$scale / parameters$beta_lambda - lgamma(parameters$alpha_lambda) - parameters$alpha_lambda * log(parameters$beta_lambda))
      # p(a | lambda)
      lb <- lb - 0.5 * sum(as.vector(lambda$shape * lambda$scale) * diag(atimesaT.mean)) - 0.5 * (D * log2pi - sum(log(lambda$shape * lambda$scale)))
      # p(G | a, Km)
      lb <- lb - 0.5 * sum(diag(GtimesGT.mean)) + crossprod(a$mean, KmtimesGT.mean) - 0.5 * sum(KmKm * atimesaT.mean) - 0.5 * N * P * (log2pi + 2 * log(sigmag))
      # p(gamma)
      lb <- lb + (parameters$alpha_gamma - 1) * (digamma(gamma$shape) + log(gamma$scale)) - gamma$shape * gamma$scale / parameters$beta_gamma - lgamma(parameters$alpha_gamma) - parameters$alpha_gamma * log(parameters$beta_gamma)
      # p(b | gamma)
      lb <- lb - 0.5 * gamma$shape * gamma$scale * btimesbT.mean - 0.5 * (log2pi - log(gamma$shape * gamma$scale))
      # p(omega)
      lb <- lb + sum((parameters$alpha_omega - 1) * (digamma(omega$shape) + log(omega$scale)) - omega$shape * omega$scale / parameters$beta_omega - lgamma(parameters$alpha_omega) - parameters$alpha_omega * log(parameters$beta_omega))
      # p(e | omega)
      lb <- lb - 0.5 * sum(as.vector(omega$shape * omega$scale) * diag(etimeseT.mean)) - 0.5 * (P * log2pi - sum(log(omega$shape * omega$scale)))
      # p(f | b, e, G)
      lb <- lb - 0.5 * (crossprod(f$mean, f$mean) + sum(f$covariance)) + crossprod(f$mean, crossprod(G$mean, be$mean[2:(P + 1)])) + sum(be$mean[1] * f$mean) - 0.5 * sum(etimeseT.mean * GtimesGT.mean) - sum(crossprod(G$mean, etimesb.mean)) - 0.5 * N * btimesbT.mean - 0.5 * N * log2pi

      # q(lambda)
      lb <- lb + sum(lambda$shape + log(lambda$scale) + lgamma(lambda$shape) + (1 - lambda$shape) * digamma(lambda$shape))
      # q(a)
      lb <- lb + 0.5 * (D * (log2pi + 1) + logdet(a$covariance))
      # q(G)
      lb <- lb + 0.5 * N * (P * (log2pi + 1) + logdet(G$covariance))
      # q(gamma)
      lb <- lb + gamma$shape + log(gamma$scale) + lgamma(gamma$shape) + (1 - gamma$shape) * digamma(gamma$shape)
      # q(omega)
      lb <- lb + sum(omega$shape + log(omega$scale) + lgamma(omega$shape) + (1 - omega$shape) * digamma(omega$shape))
      # q(b, e)
      lb <- lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be$covariance))
      # q(f)
      lb <- lb + 0.5 * sum(log2pi + f$covariance) + sum(log(normalization))
      
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