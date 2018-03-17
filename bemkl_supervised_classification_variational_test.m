function prediction = bemkl_supervised_classification_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);

    prediction.G.mu = zeros(P, N);
    prediction.G.sigma = zeros(P, N);
    for m = 1:P
        prediction.G.mu(m, :) = state.a.mu' * Km(:, :, m);
        prediction.G.sigma(m, :) = state.parameters.sigma_g^2 + diag(Km(:, :, m)' * state.a.sigma * Km(:, :, m));
    end

    prediction.f.mu = [ones(1, N); prediction.G.mu]' * state.be.mu;
    prediction.f.sigma = 1 + diag([ones(1, N); prediction.G.mu]' * state.be.sigma * [ones(1, N); prediction.G.mu]);

    pos = 1 - normcdf((+state.parameters.margin - prediction.f.mu) ./ prediction.f.sigma);
    neg = normcdf((-state.parameters.margin - prediction.f.mu) ./ prediction.f.sigma);
    prediction.p = pos ./ (pos + neg);
end
