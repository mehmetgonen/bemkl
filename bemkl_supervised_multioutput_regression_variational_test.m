function prediction = bemkl_supervised_multioutput_regression_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);
    L = length(state.be.mu) - P;    
    
    prediction.G.mu = zeros(P, N, L);
    prediction.G.sigma = zeros(P, N, L);
    for o = 1:L
        for m = 1:P
            prediction.G.mu(m, :, o) = state.A.mu(:, o)' * Km(:, :, m);
            prediction.G.sigma(m, :, o) = 1 / (state.upsilon.alpha(o) * state.upsilon.beta(o)) + diag(Km(:, :, m)' * state.A.sigma(:, :, o) * Km(:, :, m));
        end
    end   
    
    prediction.Y.mu = zeros(L, N);
    prediction.Y.sigma = zeros(L, N);
    for o = 1:L
        prediction.Y.mu(o, :) = [ones(1, N); prediction.G.mu(:, :, o)]' * state.be.mu([o, L + 1:L + P]);
        prediction.Y.sigma(o, :) = 1 / (state.epsilon.alpha(o) * state.epsilon.beta(o)) + diag([ones(1, N); prediction.G.mu(:, :, o)]' * state.be.sigma([o, L + 1:L + P], [o, L + 1:L + P]) * [ones(1, N); prediction.G.mu(:, :, o)]);
    end
end
