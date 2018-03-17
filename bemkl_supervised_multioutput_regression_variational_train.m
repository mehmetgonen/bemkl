function state = bemkl_supervised_multioutput_regression_variational_train(Km, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    L = size(Y, 1);

    log2pi = log(2 * pi);

    Lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(D, L);
    Lambda.beta = parameters.beta_lambda * ones(D, L);
    upsilon.alpha = (parameters.alpha_upsilon + 0.5 * N * P) * ones(L, 1);
    upsilon.beta = parameters.beta_upsilon * ones(L, 1);
    A.mu = randn(D, L);
    A.sigma = repmat(eye(D, D), [1, 1, L]);
    G.mu = randn(P, N, L);
    G.sigma = repmat(eye(P, P), [1, 1, L]);
    gamma.alpha = (parameters.alpha_gamma + 0.5) * ones(L, 1);
    gamma.beta = parameters.beta_gamma * ones(L, 1);
    omega.alpha = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.beta = parameters.beta_omega * ones(P, 1);
    epsilon.alpha = (parameters.alpha_epsilon + 0.5 * N) * ones(L, 1);
    epsilon.beta = parameters.beta_epsilon * ones(L, 1);
    be.mu = [zeros(L, 1); ones(P, 1)];
    be.sigma = eye(L + P, L + P);

    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    atimesaT.mu = zeros(D, D, L);
    for o = 1:L
        atimesaT.mu(:, :, o) = A.mu(:, o) * A.mu(:, o)' + A.sigma(:, :, o);
    end
    GtimesGT.mu = zeros(P, P, L);
    for o = 1:L
        GtimesGT.mu(:, :, o) = G.mu(:, :, o) * G.mu(:, :, o)' + N * G.sigma(:, :, o);
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
        %%%% update upsilon
        for o = 1:L
            upsilon.beta(o) = 1 / (1 / parameters.beta_upsilon + 0.5 * (sum(diag(GtimesGT.mu(:, :, o))) - 2 * sum(sum(reshape(A.mu(:, o)' * Km, [N, P])' .* G.mu(:, :, o))) + sum(sum(KmKm .* atimesaT.mu(:, :, o)))));
        end
        %%%% update A
        for o = 1:L
            A.sigma(:, :, o) = (diag(Lambda.alpha(:, o) .* Lambda.beta(:, o)) + upsilon.alpha(o) * upsilon.beta(o) * KmKm) \ eye(D, D);
            A.mu(:, o) = A.sigma(:, :, o) * (upsilon.alpha(o) * upsilon.beta(o) * KmtimesGT.mu(:, o));
            atimesaT.mu(:, :, o) = A.mu(:, o) * A.mu(:, o)' + A.sigma(:, :, o);
        end
        %%%% update G
        for o = 1:L
            G.sigma(:, :, o) = (upsilon.alpha(o) * upsilon.beta(o) * eye(P, P) + epsilon.alpha(o) * epsilon.beta(o) * etimeseT.mu) \ eye(P, P);
            G.mu(:, :, o) = G.sigma(:, :, o) * (upsilon.alpha(o) * upsilon.beta(o) * reshape(A.mu(:, o)' * Km, [N, P])' + epsilon.alpha(o) * epsilon.beta(o) * (be.mu(L + 1:L + P) * Y(o, :) - repmat(etimesb.mu(:, o), 1, N)));
            GtimesGT.mu(:, :, o) = G.mu(:, :, o) * G.mu(:, :, o)' + N * G.sigma(:, :, o);
            KmtimesGT.mu(:, o) = Km * reshape(G.mu(:, :, o)', N * P, 1);
        end        
        %%%% update gamma
        gamma.beta = 1 ./ (1 / parameters.beta_gamma + 0.5 * diag(btimesbT.mu));
        %%%% update omega
        omega.beta = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mu));
        %%%% update epsilon
        for o = 1:L
            epsilon.beta(o) = 1 / (1 / parameters.beta_epsilon + 0.5 * (Y(o, :) * Y(o, :)' - 2 * Y(o, :) * [ones(1, N); G.mu(:, :, o)]' * be.mu([o, L + 1:L + P]) + N * btimesbT.mu(o, o) + sum(sum(GtimesGT.mu(:, :, o) .* etimeseT.mu)) + 2 * sum(G.mu(:, :, o), 2)' * etimesb.mu(:, o)));
        end
        %%%% update b and e
        be.sigma = [diag(gamma.alpha .* gamma.beta) + N * diag(epsilon.alpha .* epsilon.beta), repmat(epsilon.alpha .* epsilon.beta, 1, P) .* squeeze(sum(G.mu, 2))'; repmat((epsilon.alpha .* epsilon.beta)', P, 1) .* squeeze(sum(G.mu, 2)), diag(omega.alpha .* omega.beta)];
        for o = 1:L
            be.sigma(L + 1:L + P, L + 1:L + P) = be.sigma(L + 1:L + P, L + 1:L + P) + epsilon.alpha(o) * epsilon.beta(o) * GtimesGT.mu(:, :, o);
        end
        be.sigma = be.sigma \ eye(L + P, L + P);
        be.mu = zeros(L + P, 1);
        be.mu(1:L) = (epsilon.alpha .* epsilon.beta) .* sum(Y, 2);
        for o = 1:L
            be.mu(L + 1:L + P) = be.mu(L + 1:L + P) + epsilon.alpha(o) * epsilon.beta(o) * G.mu(:, :, o) * Y(o, :)';
        end
        be.mu = be.sigma * be.mu;
        btimesbT.mu = be.mu(1:L) * be.mu(1:L)' + be.sigma(1:L, 1:L);
        etimeseT.mu = be.mu(L + 1:L + P) * be.mu(L + 1:L + P)' + be.sigma(L + 1:L + P, L + 1:L + P);
        for o = 1:L
            etimesb.mu(:, o) = be.mu(L + 1:L + P) * be.mu(o) + be.sigma(L + 1:L + P, o);
        end
        
        if parameters.progress == 1
            lb = 0;

            %%%% p(Lambda)
            lb = lb + sum(sum((parameters.alpha_lambda - 1) * (psi(Lambda.alpha) + log(Lambda.beta)) - Lambda.alpha .* Lambda.beta / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda)));
            %%%% p(upsilon)
            lb = lb + sum((parameters.alpha_upsilon - 1) * (psi(upsilon.alpha) + log(upsilon.beta)) - upsilon.alpha .* upsilon.beta / parameters.beta_upsilon - gammaln(parameters.alpha_upsilon) - parameters.alpha_upsilon * log(parameters.beta_upsilon));
            %%%% p(A | Lambda)
            for o = 1:L
                lb = lb - 0.5 * sum(Lambda.alpha(:, o) .* Lambda.beta(:, o) .* diag(atimesaT.mu(:, :, o))) - 0.5 * (D * log2pi - sum(psi(Lambda.alpha(:, o)) + log(Lambda.beta(:, o))));
            end
            %%%% p(G | A, Km, upsilon)
            for o = 1:L
                lb = lb - 0.5 * sum(diag(GtimesGT.mu(:, :, o))) * upsilon.alpha(o) * upsilon.beta(o) + (A.mu(:, o)' * KmtimesGT.mu(:, o)) * upsilon.alpha(o) * upsilon.beta(o) - 0.5 * sum(sum(KmKm .* atimesaT.mu(:, :, o))) * upsilon.alpha(o) * upsilon.beta(o) - 0.5 * N * P * (log2pi - (psi(upsilon.alpha(o)) + log(upsilon.beta(o))));
            end
            %%%% p(gamma)
            lb = lb + sum((parameters.alpha_gamma - 1) * (psi(gamma.alpha) + log(gamma.beta)) - gamma.alpha .* gamma.beta / parameters.beta_gamma - gammaln(parameters.alpha_gamma) - parameters.alpha_gamma * log(parameters.beta_gamma));
            %%%% p(b | gamma)
            lb = lb - 0.5 * sum(gamma.alpha .* gamma.beta .* diag(btimesbT.mu)) - 0.5 * (L * log2pi - sum(psi(gamma.alpha) + log(gamma.beta)));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.alpha) + log(omega.beta)) - omega.alpha .* omega.beta / parameters.beta_omega - gammaln(parameters.alpha_omega) - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(omega.alpha .* omega.beta .* diag(etimeseT.mu)) - 0.5 * (P * log2pi - sum(psi(omega.alpha) + log(omega.beta)));
            %%%% p(epsilon)
            lb = lb + sum((parameters.alpha_epsilon - 1) * (psi(epsilon.alpha) + log(epsilon.beta)) - epsilon.alpha .* epsilon.beta / parameters.beta_epsilon - gammaln(parameters.alpha_epsilon) - parameters.alpha_epsilon * log(parameters.beta_epsilon));
            %%%% p(Y | b, e, G, epsilon)
            for o = 1:L
                lb = lb - 0.5 * (Y(o, :) * Y(o, :)') * epsilon.alpha(o) * epsilon.beta(o) + (Y(o, :) * (G.mu(:, :, o)' * be.mu(L + 1:L + P))) * epsilon.alpha(o) * epsilon.beta(o) + sum(be.mu(o) * Y(o, :)) * epsilon.alpha(o) * epsilon.beta(o) - 0.5 * sum(sum(etimeseT.mu .* GtimesGT.mu(:, :, o))) * epsilon.alpha(o) * epsilon.beta(o) - sum(G.mu(:, :, o)' * etimesb.mu(:, o)) * epsilon.alpha(o) * epsilon.beta(o) - 0.5 * N * btimesbT.mu(o, o) * epsilon.alpha(o) * epsilon.beta(o) - 0.5 * N * (log2pi - (psi(epsilon.alpha(o)) + log(epsilon.beta(o))));
            end

            %%%% q(Lambda)
            lb = lb + sum(sum(Lambda.alpha + log(Lambda.beta) + gammaln(Lambda.alpha) + (1 - Lambda.alpha) .* psi(Lambda.alpha)));
            %%%% q(upsilon)
            lb = lb + sum(upsilon.alpha + log(upsilon.beta) + gammaln(upsilon.alpha) + (1 - upsilon.alpha) .* psi(upsilon.alpha));
            %%%% q(A)
            for o = 1:L
                lb = lb + 0.5 * (D * (log2pi + 1) + logdet(A.sigma(:, :, o)));
            end
            %%%% q(G)
            for o = 1:L
                lb = lb + 0.5 * N * (P * (log2pi + 1) + logdet(G.sigma(:, :, o)));
            end
            %%%% q(gamma)
            lb = lb + sum(gamma.alpha + log(gamma.beta) + gammaln(gamma.alpha) + (1 - gamma.alpha) .* psi(gamma.alpha));
            %%%% q(omega)
            lb = lb + sum(omega.alpha + log(omega.beta) + gammaln(omega.alpha) + (1 - omega.alpha) .* psi(omega.alpha));
            %%%% q(epsilon)
            lb = lb + sum(epsilon.alpha + log(epsilon.beta) + gammaln(epsilon.alpha) + (1 - epsilon.alpha) .* psi(epsilon.alpha));
            %%%% q(b, e)
            lb = lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be.sigma));

            bounds(iter) = lb;
        end
    end

    state.Lambda = Lambda;
    state.upsilon = upsilon;
    state.A = A;
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
