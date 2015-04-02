% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_classification_variational_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    sigma_g = parameters.sigma_g;

    log2pi = log(2 * pi);

    lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, 1);
    lambda.beta = parameters.beta_lambda * ones(D, 1);
    a.mu = randn(D, 1);
    a.sigma = eye(D, D);
    G.mu = (abs(randn(P, N)) + parameters.margin) .* sign(repmat(y', P, 1));
    G.sigma = eye(P, P);
    gamma.alpha = (parameters.alpha_gamma + 0.5);
    gamma.beta = parameters.beta_gamma;
    omega.alpha = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.beta = parameters.beta_omega * ones(P, 1);
    be.mu = [0; ones(P, 1)];
    be.sigma = eye(P + 1, P + 1);
    f.mu = (abs(randn(N, 1)) + parameters.margin) .* sign(y);
    f.sigma = ones(N, 1);

    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    lower = -1e40 * ones(N, 1);
    lower(y > 0) = +parameters.margin;
    upper = +1e40 * ones(N, 1);
    upper(y < 0) = -parameters.margin;

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
        %%%% update a
        a.sigma = (diag(lambda.alpha .* lambda.beta) + KmKm / sigma_g^2) \ eye(D, D);
        a.mu = a.sigma * KmtimesGT.mu / sigma_g^2;
        atimesaT.mu = a.mu * a.mu' + a.sigma;
        %%%% update G
        G.sigma = (eye(P, P) / sigma_g^2 + etimeseT.mu) \ eye(P, P);
        G.mu = G.sigma * (reshape(a.mu' * Km, [N, P])' / sigma_g^2 + be.mu(2:P + 1) * f.mu' - repmat(etimesb.mu, 1, N));
        GtimesGT.mu = G.mu * G.mu' + N * G.sigma;
        KmtimesGT.mu = Km * reshape(G.mu', N * P, 1);
        %%%% update gamma
        gamma.beta = 1 / (1 / parameters.beta_gamma + 0.5 * btimesbT.mu);
        %%%% update omega
        omega.beta = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mu));
        %%%% update b and e
        be.sigma = [gamma.alpha * gamma.beta + N, sum(G.mu, 2)'; ...
                         sum(G.mu, 2), diag(omega.alpha .* omega.beta) + GtimesGT.mu] \ eye(P + 1, P + 1);
        be.mu = be.sigma * ([ones(1, N); G.mu] * f.mu);
        btimesbT.mu = be.mu(1)^2 + be.sigma(1, 1);
        etimeseT.mu = be.mu(2:P + 1) * be.mu(2:P + 1)' + be.sigma(2:P + 1, 2:P + 1);
        etimesb.mu = be.mu(2:P + 1) * be.mu(1) + be.sigma(2:P + 1, 1);
        %%%% update f
        output = [ones(1, N); G.mu]' * be.mu;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        f.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        f.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        if parameters.progress == 1
            lb = 0;

            %%%% p(lambda)
            lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.alpha) + log(lambda.beta)) ...
                          - lambda.alpha .* lambda.beta / parameters.beta_lambda ...
                          - gammaln(parameters.alpha_lambda) ...
                          - parameters.alpha_lambda * log(parameters.beta_lambda));
            %%%% p(a | lambda)
            lb = lb - 0.5 * sum(lambda.alpha .* lambda.beta .* diag(atimesaT.mu)) ...
                    - 0.5 * (D * log2pi - sum(psi(lambda.alpha) + log(lambda.beta)));
            %%%% p(G | a, Km)
            lb = lb - 0.5 * sigma_g^-2 * sum(diag(GtimesGT.mu)) ...
                    + sigma_g^-2 * a.mu' * KmtimesGT.mu ...
                    - 0.5 * sigma_g^-2 * sum(sum(KmKm .* atimesaT.mu)) ...
                    - 0.5 * N * P * (log2pi + 2 * log(sigma_g));
            %%%% p(gamma)
            lb = lb + (parameters.alpha_gamma - 1) * (psi(gamma.alpha) + log(gamma.beta)) ...
                    - gamma.alpha * gamma.beta / parameters.beta_gamma ...
                    - gammaln(parameters.alpha_gamma) ...
                    - parameters.alpha_gamma * log(parameters.beta_gamma);
            %%%% p(b | gamma)
            lb = lb - 0.5 * gamma.alpha * gamma.beta * btimesbT.mu ...
                    - 0.5 * (log2pi - (psi(gamma.alpha) + log(gamma.beta)));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.alpha) + log(omega.beta)) ...
                          - omega.alpha .* omega.beta / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(omega.alpha .* omega.beta .* diag(etimeseT.mu)) ...
                    - 0.5 * (P * log2pi - sum(psi(omega.alpha) + log(omega.beta)));
            %%%% p(f | b, e, G)
            lb = lb - 0.5 * (f.mu' * f.mu + sum(f.sigma)) ...
                    + f.mu' * (G.mu' * be.mu(2:P + 1)) ...
                    + sum(be.mu(1) * f.mu) ...
                    - 0.5 * sum(sum(etimeseT.mu .* GtimesGT.mu)) ...
                    - sum(G.mu' * etimesb.mu) ...
                    - 0.5 * N * btimesbT.mu ...
                    - 0.5 * N * log2pi;

            %%%% q(lambda)
            lb = lb + sum(lambda.alpha + log(lambda.beta) + gammaln(lambda.alpha) + (1 - lambda.alpha) .* psi(lambda.alpha));
            %%%% q(a)
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(a.sigma));
            %%%% q(G)
            lb = lb + 0.5 * N * (P * (log2pi + 1) + logdet(G.sigma));
            %%%% q(gamma)
            lb = lb + gamma.alpha + log(gamma.beta) + gammaln(gamma.alpha) + (1 - gamma.alpha) * psi(gamma.alpha);
            %%%% q(omega)
            lb = lb + sum(omega.alpha + log(omega.beta) + gammaln(omega.alpha) + (1 - omega.alpha) .* psi(omega.alpha));
            %%%% q(b, e)
            lb = lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be.sigma)); 
            %%%% q(f)
            lb = lb + 0.5 * sum(log2pi + f.sigma) + sum(log(normalization));

            bounds(iter) = lb;
        end
    end

    state.lambda = lambda;
    state.a = a;
    state.gamma = gamma;
    state.omega = omega;
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