% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_multilabel_classification_variational_train(Km, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    L = size(Y, 1);
    sigmag = parameters.sigmag;

    log2pi = log(2 * pi);

    Lambda.shape = (parameters.alpha_lambda + 0.5) * ones(D, L);
    Lambda.scale = parameters.beta_lambda * ones(D, L);
    A.mean = randn(D, L);
    A.covariance = repmat(eye(D, D), [1, 1, L]);
    G.mean = (abs(randn(P, N, L)) + parameters.margin);
    for m = 1:P
        G.mean(m, :, :) = reshape(G.mean(m, :, :), N, L) .* sign(Y');
    end
    G.covariance = eye(P, P);
    gamma.shape = (parameters.alpha_gamma + 0.5) * ones(L, 1);
    gamma.scale = parameters.beta_gamma * ones(L, 1);
    omega.shape = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.scale = parameters.beta_omega * ones(P, 1);
    be.mean = [zeros(L, 1); ones(P, 1)];
    be.covariance = eye(L + P, L + P);
    F.mean = (abs(randn(L, N)) + parameters.margin) .* sign(Y);
    F.covariance = ones(L, N);
    
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

    atimesaT.mean = zeros(D, D, L);
    for o = 1:L
        atimesaT.mean(:, :, o) = A.mean(:, o) * A.mean(:, o)' + A.covariance(:, :, o);
    end
    GtimesGT.mean = zeros(P, P, L);
    for o = 1:L
        GtimesGT.mean(:, :, o) = G.mean(:, :, o) * G.mean(:, :, o)' + N * G.covariance;
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
        %%%% update A
        for o = 1:L
            A.covariance(:, :, o) = (diag(Lambda.shape(:, o) .* Lambda.scale(:, o)) + KmKm / sigmag^2) \ eye(D, D);
            A.mean(:, o) = A.covariance(:, :, o) * KmtimesGT.mean(:, o) / sigmag^2;
            atimesaT.mean(:, :, o) = A.mean(:, o) * A.mean(:, o)' + A.covariance(:, :, o);
        end
        %%%% update G
        G.covariance = (eye(P, P) / sigmag^2 + etimeseT.mean) \ eye(P, P);
        for o = 1:L
            G.mean(:, :, o) = G.covariance * (reshape(A.mean(:, o)' * Km, [N, P])' / sigmag^2 + be.mean(L + 1:L + P) * F.mean(o, :) - repmat(etimesb.mean(:, o), 1, N));
            GtimesGT.mean(:, :, o) = G.mean(:, :, o) * G.mean(:, :, o)' + N * G.covariance;
            KmtimesGT.mean(:, o) = Km * reshape(G.mean(:, :, o)', N * P, 1);
        end
        %%%% update gamma
        gamma.scale = 1 ./ (1 / parameters.beta_gamma + 0.5 * diag(btimesbT.mean));
        %%%% update omega
        omega.scale = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mean));
        %%%% update b and e
        be.covariance = [diag(gamma.shape .* gamma.scale) + N * eye(L, L), squeeze(sum(G.mean, 2))'; ...
                         squeeze(sum(G.mean, 2)), diag(omega.shape .* omega.scale)];
        for o = 1:L
            be.covariance(L + 1:L + P, L + 1:L + P) = be.covariance(L + 1:L + P, L + 1:L + P) + GtimesGT.mean(:, :, o);
        end
        be.covariance = be.covariance \ eye(L + P, L + P);
        be.mean = zeros(L + P, 1);
        be.mean(1:L) = sum(F.mean, 2);
        for o = 1:L
            be.mean(L + 1:L + P) = be.mean(L + 1:L + P) + G.mean(:, :, o) * F.mean(o, :)';
        end
        be.mean = be.covariance * be.mean;
        btimesbT.mean = be.mean(1:L) * be.mean(1:L)' + be.covariance(1:L, 1:L);
        etimeseT.mean = be.mean(L + 1:L + P) * be.mean(L + 1:L + P)' + be.covariance(L + 1:L + P, L + 1:L + P);
        for o = 1:L
            etimesb.mean(:, o) = be.mean(L + 1:L + P) * be.mean(o) + be.covariance(L + 1:L + P, o);
        end
        %%%% update F
        output = zeros(L, N);
        for o = 1:L
            output(o, :) = [ones(1, N); G.mean(:, :, o)]' * be.mean([o, L + 1:L + P]);
        end
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        F.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        F.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        if parameters.progress == 1
            lb = 0;

            %%%% p(Lambda)
            lb = lb + sum(sum((parameters.alpha_lambda - 1) * (psi(Lambda.shape) + log(Lambda.scale)) ...
                              - Lambda.shape .* Lambda.scale / parameters.beta_lambda ...
                              - gammaln(parameters.alpha_lambda) ...
                              - parameters.alpha_lambda * log(parameters.beta_lambda)));
            %%%% p(A | Lambda)
            for o = 1:L
                lb = lb - 0.5 * sum(Lambda.shape(:, o) .* Lambda.scale(:, o) .* diag(atimesaT.mean(:, :, o))) ...
                        - 0.5 * (D * log2pi - sum(log(Lambda.shape(:, o) .* Lambda.scale(:, o))));
            end
            %%%% p(G | A, Km)
            for o = 1:L
                lb = lb - 0.5 * sum(diag(GtimesGT.mean(:, :, o))) ...
                        + A.mean(:, o)' * KmtimesGT.mean(:, o) ...
                        - 0.5 * sum(sum(KmKm .* atimesaT.mean(:, :, o))) ...
                        - 0.5 * N * P * (log2pi + 2 * log(sigmag));
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
            %%%% p(F | b, e, G)
            for o = 1:L
                lb = lb - 0.5 * (F.mean(o, :) * F.mean(o, :)' + sum(F.covariance(o, :))) ...
                        + F.mean(o, :) * (G.mean(:, :, o)' * be.mean(L + 1:L + P)) ...
                        + sum(be.mean(o) * F.mean(o, :)) ...
                        - 0.5 * sum(sum(etimeseT.mean .* GtimesGT.mean(:, :, o))) ...
                        - sum(G.mean(:, :, o)' * etimesb.mean(:, o)) ...
                        - 0.5 * N * btimesbT.mean(o, o) ...
                        - 0.5 * N * log2pi;
            end

            %%%% q(Lambda)
            lb = lb + sum(sum(Lambda.shape + log(Lambda.scale) + gammaln(Lambda.shape) + (1 - Lambda.shape) .* psi(Lambda.shape)));
            %%%% q(A)
            for o = 1:L
                lb = lb + 0.5 * (D * (log2pi + 1) + logdet(A.covariance(:, :, o)));
            end
            %%%% q(G)
            lb = lb + 0.5 * L * (N * (P * (log2pi + 1) + logdet(G.covariance)));
            %%%% q(gamma)
            lb = lb + sum(gamma.shape + log(gamma.scale) + gammaln(gamma.shape) + (1 - gamma.shape) .* psi(gamma.shape));
            %%%% q(omega)
            lb = lb + sum(omega.shape + log(omega.scale) + gammaln(omega.shape) + (1 - omega.shape) .* psi(omega.shape));
            %%%% q(b, e)
            lb = lb + 0.5 * ((L + P) * (log2pi + 1) + logdet(be.covariance));
            %%%% q(F)
            lb = lb + 0.5 * sum(sum(log2pi + F.covariance)) + sum(sum(log(normalization)));

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