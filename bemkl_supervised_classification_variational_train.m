% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bemkl_supervised_classification_variational_train(Km, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(Km, 1);
    N = size(Km, 2);
    P = size(Km, 3);
    sigmag = parameters.sigmag;

    log2pi = log(2 * pi);

    lambda.shape = (parameters.alpha_lambda + 0.5) * ones(D, 1);
    lambda.scale = parameters.beta_lambda * ones(D, 1);
    a.mean = randn(D, 1);
    a.covariance = eye(D, D);
    G.mean = (abs(randn(P, N)) + parameters.margin) .* sign(repmat(y', P, 1));
    G.covariance = eye(P, P);
    gamma.shape = (parameters.alpha_gamma + 0.5);
    gamma.scale = parameters.beta_gamma;
    omega.shape = (parameters.alpha_omega + 0.5) * ones(P, 1);
    omega.scale = parameters.beta_omega * ones(P, 1);
    be.mean = [0; ones(P, 1)];
    be.covariance = eye(P + 1, P + 1);
    f.mean = (abs(randn(N, 1)) + parameters.margin) .* sign(y);
    f.covariance = ones(N, 1);

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
        %%%% update a
        a.covariance = (diag(lambda.shape .* lambda.scale) + KmKm / sigmag^2) \ eye(D, D);
        a.mean = a.covariance * KmtimesGT.mean / sigmag^2;
        atimesaT.mean = a.mean * a.mean' + a.covariance;
        %%%% update G
        G.covariance = (eye(P, P) / sigmag^2 + etimeseT.mean) \ eye(P, P);
        G.mean = G.covariance * (reshape(a.mean' * Km, [N, P])' / sigmag^2 + be.mean(2:P + 1) * f.mean' - repmat(etimesb.mean, 1, N));
        GtimesGT.mean = G.mean * G.mean' + N * G.covariance;
        KmtimesGT.mean = Km * reshape(G.mean', N * P, 1);
        %%%% update gamma
        gamma.scale = 1 / (1 / parameters.beta_gamma + 0.5 * btimesbT.mean);
        %%%% update omega
        omega.scale = 1 ./ (1 / parameters.beta_omega + 0.5 * diag(etimeseT.mean));
        %%%% update b and e
        be.covariance = [gamma.shape * gamma.scale + N, sum(G.mean, 2)'; ...
                         sum(G.mean, 2), diag(omega.shape .* omega.scale) + GtimesGT.mean] \ eye(P + 1, P + 1);
        be.mean = be.covariance * ([ones(1, N); G.mean] * f.mean);
        btimesbT.mean = be.mean(1)^2 + be.covariance(1, 1);
        etimeseT.mean = be.mean(2:P + 1) * be.mean(2:P + 1)' + be.covariance(2:P + 1, 2:P + 1);
        etimesb.mean = be.mean(2:P + 1) * be.mean(1) + be.covariance(2:P + 1, 1);
        %%%% update f
        output = [ones(1, N); G.mean]' * be.mean;
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        f.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        f.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        if parameters.progress == 1
            lb = 0;

            %%%% p(lambda)
            lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.shape) + log(lambda.scale)) ...
                          - lambda.shape .* lambda.scale / parameters.beta_lambda ...
                          - gammaln(parameters.alpha_lambda) ...
                          - parameters.alpha_lambda * log(parameters.beta_lambda));
            %%%% p(a | lambda)
            lb = lb - 0.5 * sum(lambda.shape .* lambda.scale * diag(atimesaT.mean)) ...
                    - 0.5 * (D * log2pi - sum(log(lambda.shape .* lambda.scale)));
            %%%% p(G | a, Km)
            lb = lb - 0.5 * sum(diag(GtimesGT.mean)) ...
                    + a.mean' * KmtimesGT.mean ...
                    - 0.5 * sum(sum(KmKm .* atimesaT.mean)) ...
                    - 0.5 * N * P * (log2pi + 2 * log(sigmag));
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
            %%%% p(f | b, e, G)
            lb = lb - 0.5 * (f.mean' * f.mean + sum(f.covariance)) ...
                    + f.mean' * (G.mean' * be.mean(2:P + 1)) ...
                    + sum(be.mean(1) * f.mean) ...
                    - 0.5 * sum(sum(etimeseT.mean .* GtimesGT.mean)) ...
                    - sum(G.mean' * etimesb.mean) ...
                    - 0.5 * N * btimesbT.mean ...
                    - 0.5 * N * log2pi;

            %%%% q(lambda)
            lb = lb + sum(lambda.shape + log(lambda.scale) + gammaln(lambda.shape) + (1 - lambda.shape) .* psi(lambda.shape));
            %%%% q(a)
            lb = lb + 0.5 * (D * (log2pi + 1) + logdet(a.covariance));
            %%%% q(G)
            lb = lb + 0.5 * N * (P * (log2pi + 1) + logdet(G.covariance));
            %%%% q(gamma)
            lb = lb + gamma.shape + log(gamma.scale) + gammaln(gamma.shape) + (1 - gamma.shape) * psi(gamma.shape);
            %%%% q(omega)
            lb = lb + sum(omega.shape + log(omega.scale) + gammaln(omega.shape) + (1 - omega.shape) .* psi(omega.shape));
            %%%% q(b, e)
            lb = lb + 0.5 * ((P + 1) * (log2pi + 1) + logdet(be.covariance)); 
            %%%% q(f)
            lb = lb + 0.5 * sum(log2pi + f.covariance) + sum(log(normalization));

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