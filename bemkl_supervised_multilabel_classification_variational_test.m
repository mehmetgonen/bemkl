function prediction = bemkl_supervised_multilabel_classification_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);
    L = length(state.be.mu) - P;

    prediction.G.mu = zeros(P, N, L);
    prediction.G.sigma = zeros(P, N, L);
    for o = 1:L
        for m = 1:P
            prediction.G.mu(m, :, o) = state.A.mu(:, o)' * Km(:, :, m);
            prediction.G.sigma(m, :, o) = state.parameters.sigma_g^2 + diag(Km(:, :, m)' * state.A.sigma(:, :, o) * Km(:, :, m));
        end
    end
    
    prediction.F.mu = zeros(L, N);
    prediction.F.sigma = zeros(L, N);
    for o = 1:L
        prediction.F.mu(o, :) = [ones(1, N); prediction.G.mu(:, :, o)]' * state.be.mu([o, L + 1:L + P]);
        prediction.F.sigma(o, :) = 1 + diag([ones(1, N); prediction.G.mu(:, :, o)]' * state.be.sigma([o, L + 1:L + P], [o, L + 1:L + P]) * [ones(1, N); prediction.G.mu(:, :, o)]);
    end    

    pos = 1 - normcdf((+state.parameters.margin - prediction.F.mu) ./ prediction.F.sigma);
    neg = normcdf((-state.parameters.margin - prediction.F.mu) ./ prediction.F.sigma);
    prediction.P = pos ./ (pos + neg);
end
