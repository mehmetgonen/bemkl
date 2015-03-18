% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_regression_variational_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);

    log2pi = log(2 * pi);

    lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, 1);
    lambda.beta = parameters.beta_lambda * ones(D, 1);
    upsilon.alpha = parameters.alpha_upsilon + 0.5 * N * P;
    upsilon.beta = parameters.beta_upsilon;
    a.mu = randn(D, 1);
    a.sigma = eye(D, D);
    G.mu = randn(P, N);
    G.sigma = eye(P, P);
    gamma.alpha = (parameters.alpha_gamma + 0.5);
    gamma.beta = parameters.beta_gamma;
    omega.alpha = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.beta = parameters.beta_omega * ones(P, 1);
    epsilon.alpha = parameters.alpha_epsilon + 0.5 * N;
    epsilon.beta = parameters.beta_epsilon;
    be.mu = [0; ones(P, 1)];
    be.sigma = eye(P + 1, P + 1);

    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    atimesaT.mu = a.mu * a.mu' + a.sigma;
    GtimesGT.mu = G.mu * G.mu' + N * G.sigma;
    btimesbT.mu = be.mu(1)^2 + be.sigma(1, 1);
    etimeseT.mu = be.mu(2:P + 1) * be.mu(2:P + 1)' + be.sigma(2:P + 1, 2:P + 1);
    etimesb.mu = be.mu(2:P + 1) * be.mu(1) + be.sigma(2:P + 1, 1);
    KmtimesGT.mu = Km * reshape(G.mu', N * P, 1);
    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update lambda
        lambda.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(atimesaT.mu));
        %%%% update upsilon        
        upsilon.beta = 1 / (1 / parameters.beta_upsilon + 0.5 * (sum(diag(GtimesGT.mu)) ...
                                                                  - 2 * sum(sum(reshape(a.mu' * Km, [N, P]) .* G.mu')) ...
                                                                  + sum(sum(KmKm .* atimesaT.mu))));
        %%%% update a
        a.sigma = (diag(lambda.alpha .* lambda.beta) + upsilon.alpha * upsilon.beta * KmKm) \ eye(D, D);
        a.mu = a.sigma * (upsilon.alpha * upsilon.beta * KmtimesGT.mu);
        atimesaT.mu = a.mu * a.mu' + a.sigma;
        %%%% update G
        G.sigma = (upsilon.alpha * upsilon.beta * eye(P, P) + epsilon.alpha * epsilon.beta * etimeseT.mu) \ eye(P, P);
        G.mu = G.sigma * (upsilon.alpha * upsilon.beta * reshape(a.mu' * Km, [N, P])' + epsilon.alpha * epsilon.beta * (be.mu(2:P + 1) * y' - repmat(etimesb.mu, 1, N)));
        GtimesGT.mu = G.mu * G.mu' + N * G.sigma;
        KmtimesGT.mu = Km * reshape(G.mu', N * P, 1);
        %%%% update gamma
        gamma.beta = 1 / (1 / parameters.beta_gamma + 0.5 * btimesbT.mu);
        %%%% update omega
        omega.beta = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mu));
        %%%% update epsilon
        epsilon.beta = 1 / (1 / parameters.beta_epsilon + 0.5 * (y' * y - 2 * y' * [ones(1, N); G.mu]' * be.mu ...
                                                                  + N * btimesbT.mu ...
                                                                  + sum(sum(GtimesGT.mu .* etimeseT.mu)) ...
                                                                  + 2 * sum(G.mu, 2)' * etimesb.mu));
        %%%% update b and e
        be.sigma = [gamma.alpha * gamma.beta + epsilon.alpha * epsilon.beta * N, epsilon.alpha * epsilon.beta * sum(G.mu, 2)'; ...
                         epsilon.alpha * epsilon.beta * sum(G.mu, 2), diag(omega.alpha .* omega.beta) + epsilon.alpha * epsilon.beta * (GtimesGT.mu)] \ eye(P + 1, P + 1);
        be.mu = be.sigma * (epsilon.alpha * epsilon.beta * [ones(1, N); G.mu] * y);
        btimesbT.mu = be.mu(1)^2 + be.sigma(1, 1);
        etimeseT.mu = be.mu(2:P + 1) * be.mu(2:P + 1)' + be.sigma(2:P + 1, 2:P + 1);
        etimesb.mu = be.mu(2:P + 1) * be.mu(1) + be.sigma(2:P + 1, 1);

        if parameters.progress == 1
            lb = 0;

            %%%% p(lambda)
            lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.alpha) + log(lambda.beta)) ...
                          - lambda.alpha .* lambda.beta / parameters.beta_lambda ...
                          - gammaln(parameters.alpha_lambda) ...
                          - parameters.alpha_lambda * log(parameters.beta_lambda));
            %%%% p(upsilon)
            lb = lb + (parameters.alpha_upsilon - 1) * (psi(upsilon.alpha) + log(upsilon.beta)) ...
                    - upsilon.alpha * upsilon.beta / parameters.beta_upsilon ...
                    - gammaln(parameters.alpha_upsilon) ...
                    - parameters.alpha_upsilon * log(parameters.beta_upsilon);
            %%%% p(a | lambda)
            lb = lb - 0.5 * sum(lambda.alpha .* lambda.beta .* diag(atimesaT.mu)) ...
                    - 0.5 * (D * log2pi - sum(log(lambda.alpha .* lambda.beta)));
            %%%% p(G | a, Km, upsilon)
            lb = lb - 0.5 * sum(diag(GtimesGT.mu)) * upsilon.alpha * upsilon.beta ...
                    + (a.mu' * KmtimesGT.mu) * upsilon.alpha * upsilon.beta ...
                    - 0.5 * sum(sum(KmKm .* atimesaT.mu)) * upsilon.alpha * upsilon.beta ...
                    - 0.5 * N * P * (log2pi - log(upsilon.alpha * upsilon.beta));
            %%%% p(gamma)
            lb = lb + (parameters.alpha_gamma - 1) * (psi(gamma.alpha) + log(gamma.beta)) ...
                    - gamma.alpha * gamma.beta / parameters.beta_gamma ...
                    - gammaln(parameters.alpha_gamma) ...
                    - parameters.alpha_gamma * log(parameters.beta_gamma);
            %%%% p(b | gamma)
            lb = lb - 0.5 * gamma.alpha * gamma.beta * btimesbT.mu ...
                    - 0.5 * (log2pi - log(gamma.alpha * gamma.beta));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.alpha) + log(omega.beta)) ...
                          - omega.alpha .* omega.beta / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(omega.alpha .* omega.beta .* diag(etimeseT.mu)) ...
                    - 0.5 * (P * log2pi - sum(log(omega.alpha .* omega.beta)));
            %%%% p(epsilon)
            lb = lb + (parameters.alpha_epsilon - 1) * (psi(epsilon.alpha) + log(epsilon.beta)) ...
                    - epsilon.alpha * epsilon.beta / parameters.beta_epsilon ...
                    - gammaln(parameters.alpha_epsilon) ...
                    - parameters.alpha_epsilon * log(parameters.beta_epsilon);
            %%%% p(y | b, e, G, epsilon)
            lb = lb - 0.5 * (y' * y) * epsilon.alpha * epsilon.beta ...
                    + (y' * (G.mu' * be.mu(2:P + 1))) * epsilon.alpha * epsilon.beta ...
                    + sum(be.mu(1) * y) * epsilon.alpha * epsilon.beta ...
                    - 0.5 * sum(sum(etimeseT.mu .* GtimesGT.mu)) * epsilon.alpha * epsilon.beta ...
                    - sum(G.mu' * etimesb.mu) * epsilon.alpha * epsilon.beta ...
                    - 0.5 * N * btimesbT.mu * epsilon.alpha * epsilon.beta ...
                    - 0.5 * N * (log2pi - log(epsilon.alpha * epsilon.beta));

            %%%% q(lambda)
            lb = lb + sum(lambda.alpha + log(lambda.beta) + gammaln(lambda.alpha) + (1 - lambda.alpha) .* psi(lambda.alpha));
            %%%% q(upsilon)
            lb = lb + upsilon.alpha + log(upsilon.beta) + gammaln(upsilon.alpha) + (1 - upsilon.alpha) * psi(upsilon.alpha);
            %%%% q(a)
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(a.sigma));
            %%%% q(G)
            lb = lb + 0.5 * N * (P * (log2pi + 1) + logdet(G.sigma));
            %%%% q(gamma)
            lb = lb + gamma.alpha + log(gamma.beta) + gammaln(gamma.alpha) + (1 - gamma.alpha) * psi(gamma.alpha);
            %%%% q(omega)
            lb = lb + sum(omega.alpha + log(omega.beta) + gammaln(omega.alpha) + (1 - omega.alpha) .* psi(omega.alpha));
            %%%% q(epsilon)
            lb = lb + epsilon.alpha + log(epsilon.beta) + gammaln(epsilon.alpha) + (1 - epsilon.alpha) * psi(epsilon.alpha);
            %%%% q(b, e)
            lb = lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be.sigma));

            bounds(iter) = lb;
        end
    end

    state.lambda = lambda;
    state.upsilon = upsilon;
    state.a = a;
    state.gamma = gamma;
    state.omega = omega;
    state.epsilon = epsilon;    
    state.be = be;
    if parameters.progress == 1
        state.bounds = bounds;
    end
    state.parameters = parameters;
end

function ld = logdet(Sigma)
    U = chol(Sigma);
    ld = 2 * sum(log(diag(U)));
end