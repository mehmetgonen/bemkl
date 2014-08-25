% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_multioutput_regression_variational_train(Km, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    L = size(Y, 1);

    log2pi = log(2 * pi);

    Lambda.shape = (parameters.alpha_lambda + 0.5) * ones(D, L);
    Lambda.scale = parameters.beta_lambda * ones(D, L);
    upsilon.shape = (parameters.alpha_upsilon + 0.5 * N * P) * ones(L, 1);
    upsilon.scale = parameters.beta_upsilon * ones(L, 1);
    A.mean = randn(D, L);
    A.covariance = repmat(eye(D, D), [1, 1, L]);
    G.mean = randn(P, N, L);
    G.covariance = repmat(eye(P, P), [1, 1, L]);
    gamma.shape = (parameters.alpha_gamma + 0.5) * ones(L, 1);
    gamma.scale = parameters.beta_gamma * ones(L, 1);
    omega.shape = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.scale = parameters.beta_omega * ones(P, 1);
    epsilon.shape = (parameters.alpha_epsilon + 0.5 * N) * ones(L, 1);
    epsilon.scale = parameters.beta_epsilon * ones(L, 1);
    be.mean = [zeros(L, 1); ones(P, 1)];
    be.covariance = eye(L + P, L + P);

    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    atimesaT.mean = zeros(D, D, L);
    for o = 1:L
        atimesaT.mean(:, :, o) = A.mean(:, o) * A.mean(:, o)' + A.covariance(:, :, o);
    end
    GtimesGT.mean = zeros(P, P, L);
    for o = 1:L
        GtimesGT.mean(:, :, o) = G.mean(:, :, o) * G.mean(:, :, o)' + N * G.covariance(:, :, o);
    end
    btimesbT.mean = be.mean(1:L) * be.mean(1:L)' + be.covariance(1:L, 1:L);
    etimeseT.mean = be.mean(L + 1:L + P) * be.mean(L + 1:L + P)' + be.covariance(L + 1:L + P, L + 1:L + P);
    etimesb.mean = zeros(P, L);
    for o = 1:L
        etimesb.mean(:, o) = be.mean(L + 1:L + P) * be.mean(o) + be.covariance(L + 1:L + P, o);
    end
    KmtimesGT.mean = zeros(D, L);
    for o = 1:L
        KmtimesGT.mean(:, o) = Km * reshape(G.mean(:, :, o)', N * P, 1);
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
            Lambda.scale(:, o) = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(atimesaT.mean(:, :, o)));
        end
        %%%% update upsilon
        for o = 1:L
            upsilon.scale(o) = 1 / (1 / parameters.beta_upsilon + 0.5 * (sum(diag(GtimesGT.mean(:, :, o))) ...
                                                                         - 2 * sum(sum(reshape(A.mean(:, o)' * Km, [N, P])' .* G.mean(:, :, o))) ...
                                                                         + sum(sum(KmKm .* atimesaT.mean(:, :, o)))));
        end
        %%%% update A
        for o = 1:L
            A.covariance(:, :, o) = (diag(Lambda.shape(:, o) .* Lambda.scale(:, o)) + upsilon.shape(o) * upsilon.scale(o) * KmKm) \ eye(D, D);
            A.mean(:, o) = A.covariance(:, :, o) * (upsilon.shape(o) * upsilon.scale(o) * KmtimesGT.mean(:, o));
            atimesaT.mean(:, :, o) = A.mean(:, o) * A.mean(:, o)' + A.covariance(:, :, o);
        end
        %%%% update G
        for o = 1:L
            G.covariance(:, :, o) = (upsilon.shape(o) * upsilon.scale(o) * eye(P, P) + epsilon.shape(o) * epsilon.scale(o) * etimeseT.mean) \ eye(P, P);
            G.mean(:, :, o) = G.covariance(:, :, o) * (upsilon.shape(o) * upsilon.scale(o) * reshape(A.mean(:, o)' * Km, [N, P])' + epsilon.shape(o) * epsilon.scale(o) * (be.mean(L + 1:L + P) * Y(o, :) - repmat(etimesb.mean(:, o), 1, N)));
            GtimesGT.mean(:, :, o) = G.mean(:, :, o) * G.mean(:, :, o)' + N * G.covariance(:, :, o);
            KmtimesGT.mean(:, o) = Km * reshape(G.mean(:, :, o)', N * P, 1);
        end        
        %%%% update gamma
        gamma.scale = 1 ./ (1 / parameters.beta_gamma + 0.5 * diag(btimesbT.mean));
        %%%% update omega
        omega.scale = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mean));
        %%%% update epsilon
        for o = 1:L
            epsilon.scale(o) = 1 / (1 / parameters.beta_epsilon + 0.5 * (Y(o, :) * Y(o, :)' - 2 * Y(o, :) * [ones(1, N); G.mean(:, :, o)]' * be.mean([o, L + 1:L + P]) ...
                                                                         + N * btimesbT.mean(o, o) ...
                                                                         + sum(sum(GtimesGT.mean(:, :, o) .* etimeseT.mean)) ...
                                                                         + 2 * sum(G.mean(:, :, o), 2)' * etimesb.mean(:, o)));
        end
        %%%% update b and e
        be.covariance = [diag(gamma.shape .* gamma.scale) + N * diag(epsilon.shape .* epsilon.scale), repmat(epsilon.shape .* epsilon.scale, 1, P) .* squeeze(sum(G.mean, 2))'; ...
                         repmat((epsilon.shape .* epsilon.scale)', P, 1) .* squeeze(sum(G.mean, 2)), diag(omega.shape .* omega.scale)];
        for o = 1:L
            be.covariance(L + 1:L + P, L + 1:L + P) = be.covariance(L + 1:L + P, L + 1:L + P) + epsilon.shape(o) * epsilon.scale(o) * GtimesGT.mean(:, :, o);
        end
        be.covariance = be.covariance \ eye(L + P, L + P);
        be.mean = zeros(L + P, 1);
        be.mean(1:L) = (epsilon.shape .* epsilon.scale) .* sum(Y, 2);
        for o = 1:L
            be.mean(L + 1:L + P) = be.mean(L + 1:L + P) + epsilon.shape(o) * epsilon.scale(o) * G.mean(:, :, o) * Y(o, :)';
        end
        be.mean = be.covariance * be.mean;
        btimesbT.mean = be.mean(1:L) * be.mean(1:L)' + be.covariance(1:L, 1:L);
        etimeseT.mean = be.mean(L + 1:L + P) * be.mean(L + 1:L + P)' + be.covariance(L + 1:L + P, L + 1:L + P);
        for o = 1:L
            etimesb.mean(:, o) = be.mean(L + 1:L + P) * be.mean(o) + be.covariance(L + 1:L + P, o);
        end
        
        if parameters.progress == 1
            lb = 0;

            %%%% p(Lambda)
            lb = lb + sum(sum((parameters.alpha_lambda - 1) * (psi(Lambda.shape) + log(Lambda.scale)) ...
                              - Lambda.shape .* Lambda.scale / parameters.beta_lambda ...
                              - gammaln(parameters.alpha_lambda) ...
                              - parameters.alpha_lambda * log(parameters.beta_lambda)));
            %%%% p(upsilon)
            lb = lb + sum((parameters.alpha_upsilon - 1) * (psi(upsilon.shape) + log(upsilon.scale)) ...
                          - upsilon.shape .* upsilon.scale / parameters.beta_upsilon ...
                          - gammaln(parameters.alpha_upsilon) ...
                          - parameters.alpha_upsilon * log(parameters.beta_upsilon));
            %%%% p(A | Lambda)
            for o = 1:L
                lb = lb - 0.5 * sum(Lambda.shape(:, o) .* Lambda.scale(:, o) .* diag(atimesaT.mean(:, :, o))) ...
                        - 0.5 * (D * log2pi - sum(log(Lambda.shape(:, o) .* Lambda.scale(:, o))));
            end
            %%%% p(G | A, Km, upsilon)
            for o = 1:L
                lb = lb - 0.5 * sum(diag(GtimesGT.mean(:, :, o))) * upsilon.shape(o) * upsilon.scale(o) ...
                        + (A.mean(:, o)' * KmtimesGT.mean(:, o)) * upsilon.shape(o) * upsilon.scale(o) ...
                        - 0.5 * sum(sum(KmKm .* atimesaT.mean(:, :, o))) * upsilon.shape(o) * upsilon.scale(o) ...
                        - 0.5 * N * P * (log2pi - log(upsilon.shape(o) * upsilon.scale(o)));
            end
            %%%% p(gamma)
            lb = lb + sum((parameters.alpha_gamma - 1) * (psi(gamma.shape) + log(gamma.scale)) ...
                          - gamma.shape .* gamma.scale / parameters.beta_gamma ...
                          - gammaln(parameters.alpha_gamma) ...
                          - parameters.alpha_gamma * log(parameters.beta_gamma));
            %%%% p(b | gamma)
            lb = lb - 0.5 * sum(gamma.shape .* gamma.scale .* diag(btimesbT.mean)) ...
                    - 0.5 * (L * log2pi - sum(log(gamma.shape .* gamma.scale)));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.shape) + log(omega.scale)) ...
                          - omega.shape .* omega.scale / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(omega.shape .* omega.scale .* diag(etimeseT.mean)) ...
                    - 0.5 * (P * log2pi - sum(log(omega.shape .* omega.scale)));
            %%%% p(epsilon)
            lb = lb + sum((parameters.alpha_epsilon - 1) * (psi(epsilon.shape) + log(epsilon.scale)) ...
                          - epsilon.shape .* epsilon.scale / parameters.beta_epsilon ...
                          - gammaln(parameters.alpha_epsilon) ...
                          - parameters.alpha_epsilon * log(parameters.beta_epsilon));
            %%%% p(Y | b, e, G, epsilon)
            for o = 1:L
                lb = lb - 0.5 * (Y(o, :) * Y(o, :)') * epsilon.shape(o) * epsilon.scale(o) ...
                        + (Y(o, :) * (G.mean(:, :, o)' * be.mean(L + 1:L + P))) * epsilon.shape(o) * epsilon.scale(o) ...
                        + sum(be.mean(o) * Y(o, :)) * epsilon.shape(o) * epsilon.scale(o) ...
                        - 0.5 * sum(sum(etimeseT.mean .* GtimesGT.mean(:, :, o))) * epsilon.shape(o) * epsilon.scale(o) ...
                        - sum(G.mean(:, :, o)' * etimesb.mean(:, o)) * epsilon.shape(o) * epsilon.scale(o) ...
                        - 0.5 * N * btimesbT.mean(o, o) * epsilon.shape(o) * epsilon.scale(o) ...
                        - 0.5 * N * (log2pi - log(epsilon.shape(o) * epsilon.scale(o)));
            end

            %%%% q(Lambda)
            lb = lb + sum(sum(Lambda.shape + log(Lambda.scale) + gammaln(Lambda.shape) + (1 - Lambda.shape) .* psi(Lambda.shape)));
            %%%% q(upsilon)
            lb = lb + sum(upsilon.shape + log(upsilon.scale) + gammaln(upsilon.shape) + (1 - upsilon.shape) .* psi(upsilon.shape));
            %%%% q(A)
            for o = 1:L
                lb = lb + 0.5 * (D * (log2pi + 1) + logdet(A.covariance(:, :, o)));
            end
            %%%% q(G)
            for o = 1:L
                lb = lb + 0.5 * N * (P * (log2pi + 1) + logdet(G.covariance(:, :, o)));
            end
            %%%% q(gamma)
            lb = lb + sum(gamma.shape + log(gamma.scale) + gammaln(gamma.shape) + (1 - gamma.shape) .* psi(gamma.shape));
            %%%% q(omega)
            lb = lb + sum(omega.shape + log(omega.scale) + gammaln(omega.shape) + (1 - omega.shape) .* psi(omega.shape));
            %%%% q(epsilon)
            lb = lb + sum(epsilon.shape + log(epsilon.scale) + gammaln(epsilon.shape) + (1 - epsilon.shape) .* psi(epsilon.shape));
            %%%% q(b, e)
            lb = lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be.covariance));

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