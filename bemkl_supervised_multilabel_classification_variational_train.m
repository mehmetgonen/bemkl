% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_multilabel_classification_variational_train(Km, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    L = size(Y, 1);
    sigma_g = parameters.sigma_g;

    log2pi = log(2 * pi);

    Lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, L);
    Lambda.beta = parameters.beta_lambda * ones(D, L);
    A.mu = randn(D, L);
    A.sigma = repmat(eye(D, D), [1, 1, L]);
    G.mu = (abs(randn(P, N, L)) + parameters.margin);
    for m = 1:P
        G.mu(m, :, :) = reshape(G.mu(m, :, :), N, L) .* sign(Y');
    end
    G.sigma = eye(P, P);
    gamma.alpha = (parameters.alpha_gamma + 0.5) * ones(L, 1);
    gamma.beta = parameters.beta_gamma * ones(L, 1);
    omega.alpha = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.beta = parameters.beta_omega * ones(P, 1);
    be.mu = [zeros(L, 1); ones(P, 1)];
    be.sigma = eye(L + P, L + P);
    F.mu = (abs(randn(L, N)) + parameters.margin) .* sign(Y);
    F.sigma = ones(L, N);
    
    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    lower = -1e40 * ones(L, N);
    lower(Y > 0) = +parameters.margin;
    upper = +1e40 * ones(L, N);
    upper(Y < 0) = -parameters.margin;

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    atimesaT.mu = zeros(D, D, L);
    for o = 1:L
        atimesaT.mu(:, :, o) = A.mu(:, o) * A.mu(:, o)' + A.sigma(:, :, o);
    end
    GtimesGT.mu = zeros(P, P, L);
    for o = 1:L
        GtimesGT.mu(:, :, o) = G.mu(:, :, o) * G.mu(:, :, o)' + N * G.sigma;
    end
    btimesbT.mu = be.mu(1:L) * be.mu(1:L)' + be.sigma(1:L, 1:L);
    etimeseT.mu = be.mu(L + 1:L + P) * be.mu(L + 1:L + P)' + be.sigma(L + 1:L + P, L + 1:L + P);
    etimesb.mu = zeros(P, L);
    for o = 1:L
        etimesb.mu(:, o) = be.mu(L + 1:L + P) * be.mu(o) + be.sigma(L + 1:L + P, o);
    end
    KmtimesGT.mu = zeros(D, L);
    for o = 1:L
        KmtimesGT.mu(:, o) = Km * reshape(G.mu(:, :, o)', N * P, 1);
    end
    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Lambda
        for o = 1:L
            Lambda.beta(:, o) = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(atimesaT.mu(:, :, o)));
        end
        %%%% update A
        for o = 1:L
            A.sigma(:, :, o) = (diag(Lambda.alpha(:, o) .* Lambda.beta(:, o)) + KmKm / sigma_g^2) \ eye(D, D);
            A.mu(:, o) = A.sigma(:, :, o) * KmtimesGT.mu(:, o) / sigma_g^2;
            atimesaT.mu(:, :, o) = A.mu(:, o) * A.mu(:, o)' + A.sigma(:, :, o);
        end
        %%%% update G
        G.sigma = (eye(P, P) / sigma_g^2 + etimeseT.mu) \ eye(P, P);
        for o = 1:L
            G.mu(:, :, o) = G.sigma * (reshape(A.mu(:, o)' * Km, [N, P])' / sigma_g^2 + be.mu(L + 1:L + P) * F.mu(o, :) - repmat(etimesb.mu(:, o), 1, N));
            GtimesGT.mu(:, :, o) = G.mu(:, :, o) * G.mu(:, :, o)' + N * G.sigma;
            KmtimesGT.mu(:, o) = Km * reshape(G.mu(:, :, o)', N * P, 1);
        end
        %%%% update gamma
        gamma.beta = 1 ./ (1 / parameters.beta_gamma + 0.5 * diag(btimesbT.mu));
        %%%% update omega
        omega.beta = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mu));
        %%%% update b and e
        be.sigma = [diag(gamma.alpha .* gamma.beta) + N * eye(L, L), squeeze(sum(G.mu, 2))'; ...
                         squeeze(sum(G.mu, 2)), diag(omega.alpha .* omega.beta)];
        for o = 1:L
            be.sigma(L + 1:L + P, L + 1:L + P) = be.sigma(L + 1:L + P, L + 1:L + P) + GtimesGT.mu(:, :, o);
        end
        be.sigma = be.sigma \ eye(L + P, L + P);
        be.mu = zeros(L + P, 1);
        be.mu(1:L) = sum(F.mu, 2);
        for o = 1:L
            be.mu(L + 1:L + P) = be.mu(L + 1:L + P) + G.mu(:, :, o) * F.mu(o, :)';
        end
        be.mu = be.sigma * be.mu;
        btimesbT.mu = be.mu(1:L) * be.mu(1:L)' + be.sigma(1:L, 1:L);
        etimeseT.mu = be.mu(L + 1:L + P) * be.mu(L + 1:L + P)' + be.sigma(L + 1:L + P, L + 1:L + P);
        for o = 1:L
            etimesb.mu(:, o) = be.mu(L + 1:L + P) * be.mu(o) + be.sigma(L + 1:L + P, o);
        end
        %%%% update F
        output = zeros(L, N);
        for o = 1:L
            output(o, :) = [ones(1, N); G.mu(:, :, o)]' * be.mu([o, L + 1:L + P]);
        end
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        F.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        F.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        if parameters.progress == 1
            lb = 0;

            %%%% p(Lambda)
            lb = lb + sum(sum((parameters.alpha_lambda - 1) * (psi(Lambda.alpha) + log(Lambda.beta)) ...
                              - Lambda.alpha .* Lambda.beta / parameters.beta_lambda ...
                              - gammaln(parameters.alpha_lambda) ...
                              - parameters.alpha_lambda * log(parameters.beta_lambda)));
            %%%% p(A | Lambda)
            for o = 1:L
                lb = lb - 0.5 * sum(Lambda.alpha(:, o) .* Lambda.beta(:, o) .* diag(atimesaT.mu(:, :, o))) ...
                        - 0.5 * (D * log2pi - sum(log(Lambda.alpha(:, o) .* Lambda.beta(:, o))));
            end
            %%%% p(G | A, Km)
            for o = 1:L
                lb = lb - 0.5 * sum(diag(GtimesGT.mu(:, :, o))) ...
                        + A.mu(:, o)' * KmtimesGT.mu(:, o) ...
                        - 0.5 * sum(sum(KmKm .* atimesaT.mu(:, :, o))) ...
                        - 0.5 * N * P * (log2pi + 2 * log(sigma_g));
            end
            %%%% p(gamma)
            lb = lb + sum((parameters.alpha_gamma - 1) * (psi(gamma.alpha) + log(gamma.beta)) ...
                          - gamma.alpha .* gamma.beta / parameters.beta_gamma ...
                          - gammaln(parameters.alpha_gamma) ...
                          - parameters.alpha_gamma * log(parameters.beta_gamma));
            %%%% p(b | gamma)
            lb = lb - 0.5 * sum(gamma.alpha .* gamma.beta .* diag(btimesbT.mu)) ...
                    - 0.5 * (L * log2pi - sum(log(gamma.alpha .* gamma.beta)));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.alpha) + log(omega.beta)) ...
                          - omega.alpha .* omega.beta / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(omega.alpha .* omega.beta .* diag(etimeseT.mu)) ...
                    - 0.5 * (P * log2pi - sum(log(omega.alpha .* omega.beta)));
            %%%% p(F | b, e, G)
            for o = 1:L
                lb = lb - 0.5 * (F.mu(o, :) * F.mu(o, :)' + sum(F.sigma(o, :))) ...
                        + F.mu(o, :) * (G.mu(:, :, o)' * be.mu(L + 1:L + P)) ...
                        + sum(be.mu(o) * F.mu(o, :)) ...
                        - 0.5 * sum(sum(etimeseT.mu .* GtimesGT.mu(:, :, o))) ...
                        - sum(G.mu(:, :, o)' * etimesb.mu(:, o)) ...
                        - 0.5 * N * btimesbT.mu(o, o) ...
                        - 0.5 * N * log2pi;
            end

            %%%% q(Lambda)
            lb = lb + sum(sum(Lambda.alpha + log(Lambda.beta) + gammaln(Lambda.alpha) + (1 - Lambda.alpha) .* psi(Lambda.alpha)));
            %%%% q(A)
            for o = 1:L
                lb = lb + 0.5 * (D * (log2pi + 1) + logdet(A.sigma(:, :, o)));
            end
            %%%% q(G)
            lb = lb + 0.5 * L * (N * (P * (log2pi + 1) + logdet(G.sigma)));
            %%%% q(gamma)
            lb = lb + sum(gamma.alpha + log(gamma.beta) + gammaln(gamma.alpha) + (1 - gamma.alpha) .* psi(gamma.alpha));
            %%%% q(omega)
            lb = lb + sum(omega.alpha + log(omega.beta) + gammaln(omega.alpha) + (1 - omega.alpha) .* psi(omega.alpha));
            %%%% q(b, e)
            lb = lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be.sigma));
            %%%% q(F)
            lb = lb + 0.5 * sum(sum(log2pi + F.sigma)) + sum(sum(log(normalization)));

            bounds(iter) = lb;
        end
    end

    state.Lambda = Lambda;
    state.A = A;
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