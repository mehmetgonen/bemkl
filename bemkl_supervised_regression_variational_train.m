% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_regression_variational_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);

    log2pi = log(2 * pi);

    lambda.shape = (parameters.alpha_lambda + 0.5) * ones(D, 1);
    lambda.scale = parameters.beta_lambda * ones(D, 1);
    upsilon.shape = parameters.alpha_upsilon + 0.5 * N * P;
    upsilon.scale = parameters.beta_upsilon;
    a.mean = randn(D, 1);
    a.covariance = eye(D, D);
    G.mean = randn(P, N);
    G.covariance = eye(P, P);
    gamma.shape = (parameters.alpha_gamma + 0.5);
    gamma.scale = parameters.beta_gamma;
    omega.shape = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.scale = parameters.beta_omega * ones(P, 1);
    epsilon.shape = parameters.alpha_epsilon + 0.5 * N;
    epsilon.scale = parameters.beta_epsilon;
    be.mean = [0; ones(P, 1)];
    be.covariance = eye(P + 1, P + 1);

    KmKm = zeros(D, D);
    for m = 1:P
        KmKm = KmKm + Km(:, :, m) * Km(:, :, m)';
    end
    Km = reshape(Km, [D, N * P]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    atimesaT.mean = a.mean * a.mean' + a.covariance;
    GtimesGT.mean = G.mean * G.mean' + N * G.covariance;
    btimesbT.mean = be.mean(1)^2 + be.covariance(1, 1);
    etimeseT.mean = be.mean(2:P + 1) * be.mean(2:P + 1)' + be.covariance(2:P + 1, 2:P + 1);
    etimesb.mean = be.mean(2:P + 1) * be.mean(1) + be.covariance(2:P + 1, 1);
    KmtimesGT.mean = Km * reshape(G.mean', N * P, 1);
    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update lambda
        lambda.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * diag(atimesaT.mean));
        %%%% update upsilon        
        upsilon.scale = 1 / (1 / parameters.beta_upsilon + 0.5 * (sum(diag(GtimesGT.mean)) ...
                                                                  - 2 * sum(sum(reshape(a.mean' * Km, [N, P]) .* G.mean')) ...
                                                                  + sum(sum(KmKm .* atimesaT.mean))));
        %%%% update a
        a.covariance = (diag(lambda.shape .* lambda.scale) + upsilon.shape * upsilon.scale * KmKm) \ eye(D, D);
        a.mean = a.covariance * (upsilon.shape * upsilon.scale * KmtimesGT.mean);
        atimesaT.mean = a.mean * a.mean' + a.covariance;
        %%%% update G
        G.covariance = (upsilon.shape * upsilon.scale * eye(P, P) + epsilon.shape * epsilon.scale * etimeseT.mean) \ eye(P, P);
        G.mean = G.covariance * (upsilon.shape * upsilon.scale * reshape(a.mean' * Km, [N, P])' + epsilon.shape * epsilon.scale * (be.mean(2:P + 1) * y' - repmat(etimesb.mean, 1, N)));
        GtimesGT.mean = G.mean * G.mean' + N * G.covariance;
        KmtimesGT.mean = Km * reshape(G.mean', N * P, 1);
        %%%% update gamma
        gamma.scale = 1 / (1 / parameters.beta_gamma + 0.5 * btimesbT.mean);
        %%%% update omega
        omega.scale = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mean));
        %%%% update epsilon
        epsilon.scale = 1 / (1 / parameters.beta_epsilon + 0.5 * (y' * y - 2 * y' * [ones(1, N); G.mean]' * be.mean ...
                                                                  + N * btimesbT.mean ...
                                                                  + sum(sum(GtimesGT.mean .* etimeseT.mean)) ...
                                                                  + 2 * sum(G.mean, 2)' * etimesb.mean));
        %%%% update b and e
        be.covariance = [gamma.shape * gamma.scale + epsilon.shape * epsilon.scale * N, epsilon.shape * epsilon.scale * sum(G.mean, 2)'; ...
                         epsilon.shape * epsilon.scale * sum(G.mean, 2), diag(omega.shape .* omega.scale) + epsilon.shape * epsilon.scale * (GtimesGT.mean)] \ eye(P + 1, P + 1);
        be.mean = be.covariance * (epsilon.shape * epsilon.scale * [ones(1, N); G.mean] * y);
        btimesbT.mean = be.mean(1)^2 + be.covariance(1, 1);
        etimeseT.mean = be.mean(2:P + 1) * be.mean(2:P + 1)' + be.covariance(2:P + 1, 2:P + 1);
        etimesb.mean = be.mean(2:P + 1) * be.mean(1) + be.covariance(2:P + 1, 1);

        if parameters.progress == 1
            lb = 0;

            %%%% p(lambda)
            lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.shape) + log(lambda.scale)) ...
                          - lambda.shape .* lambda.scale / parameters.beta_lambda ...
                          - gammaln(parameters.alpha_lambda) ...
                          - parameters.alpha_lambda * log(parameters.beta_lambda));
            %%%% p(upsilon)
            lb = lb + (parameters.alpha_upsilon - 1) * (psi(upsilon.shape) + log(upsilon.scale)) ...
                    - upsilon.shape * upsilon.scale / parameters.beta_upsilon ...
                    - gammaln(parameters.alpha_upsilon) ...
                    - parameters.alpha_upsilon * log(parameters.beta_upsilon);
            %%%% p(a | lambda)
            lb = lb - 0.5 * sum(lambda.shape .* lambda.scale .* diag(atimesaT.mean)) ...
                    - 0.5 * (D * log2pi - sum(log(lambda.shape .* lambda.scale)));
            %%%% p(G | a, Km, upsilon)
            lb = lb - 0.5 * sum(diag(GtimesGT.mean)) * upsilon.shape * upsilon.scale ...
                    + (a.mean' * KmtimesGT.mean) * upsilon.shape * upsilon.scale ...
                    - 0.5 * sum(sum(KmKm .* atimesaT.mean)) * upsilon.shape * upsilon.scale ...
                    - 0.5 * N * P * (log2pi - log(upsilon.shape * upsilon.scale));
            %%%% p(gamma)
            lb = lb + (parameters.alpha_gamma - 1) * (psi(gamma.shape) + log(gamma.scale)) ...
                    - gamma.shape * gamma.scale / parameters.beta_gamma ...
                    - gammaln(parameters.alpha_gamma) ...
                    - parameters.alpha_gamma * log(parameters.beta_gamma);
            %%%% p(b | gamma)
            lb = lb - 0.5 * gamma.shape * gamma.scale * btimesbT.mean ...
                    - 0.5 * (log2pi - log(gamma.shape * gamma.scale));
            %%%% p(omega)
            lb = lb + sum((parameters.alpha_omega - 1) * (psi(omega.shape) + log(omega.scale)) ...
                          - omega.shape .* omega.scale / parameters.beta_omega ...
                          - gammaln(parameters.alpha_omega) ...
                          - parameters.alpha_omega * log(parameters.beta_omega));
            %%%% p(e | omega)
            lb = lb - 0.5 * sum(omega.shape .* omega.scale .* diag(etimeseT.mean)) ...
                    - 0.5 * (P * log2pi - sum(log(omega.shape .* omega.scale)));
            %%%% p(epsilon)
            lb = lb + (parameters.alpha_epsilon - 1) * (psi(epsilon.shape) + log(epsilon.scale)) ...
                    - epsilon.shape * epsilon.scale / parameters.beta_epsilon ...
                    - gammaln(parameters.alpha_epsilon) ...
                    - parameters.alpha_epsilon * log(parameters.beta_epsilon);
            %%%% p(y | b, e, G, epsilon)
            lb = lb - 0.5 * (y' * y) * epsilon.shape * epsilon.scale ...
                    + (y' * (G.mean' * be.mean(2:P + 1))) * epsilon.shape * epsilon.scale ...
                    + sum(be.mean(1) * y) * epsilon.shape * epsilon.scale ...
                    - 0.5 * sum(sum(etimeseT.mean .* GtimesGT.mean)) * epsilon.shape * epsilon.scale ...
                    - sum(G.mean' * etimesb.mean) * epsilon.shape * epsilon.scale ...
                    - 0.5 * N * btimesbT.mean * epsilon.shape * epsilon.scale ...
                    - 0.5 * N * (log2pi - log(epsilon.shape * epsilon.scale));

            %%%% q(lambda)
            lb = lb + sum(lambda.shape + log(lambda.scale) + gammaln(lambda.shape) + (1 - lambda.shape) .* psi(lambda.shape));
            %%%% q(upsilon)
            lb = lb + upsilon.shape + log(upsilon.scale) + gammaln(upsilon.shape) + (1 - upsilon.shape) * psi(upsilon.shape);
            %%%% q(a)
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(a.covariance));
            %%%% q(G)
            lb = lb + 0.5 * N * (P * (log2pi + 1) + logdet(G.covariance));
            %%%% q(gamma)
            lb = lb + gamma.shape + log(gamma.scale) + gammaln(gamma.shape) + (1 - gamma.shape) * psi(gamma.shape);
            %%%% q(omega)
            lb = lb + sum(omega.shape + log(omega.scale) + gammaln(omega.shape) + (1 - omega.shape) .* psi(omega.shape));
            %%%% q(epsilon)
            lb = lb + epsilon.shape + log(epsilon.scale) + gammaln(epsilon.shape) + (1 - epsilon.shape) * psi(epsilon.shape);
            %%%% q(b, e)
            lb = lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be.covariance));

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