function prediction = bemkl_supervised_regression_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);

    prediction.G.mu = zeros(P, N);
    prediction.G.sigma = zeros(P, N);
    for m = 1:P
        prediction.G.mu(m, :) = state.a.mu' * Km(:, :, m);
        prediction.G.sigma(m, :) = 1 / (state.upsilon.alpha * state.upsilon.beta) + diag(Km(:, :, m)' * state.a.sigma * Km(:, :, m));
    end
    
    prediction.y.mu = [ones(1, N); prediction.G.mu]' * state.be.mu;
    prediction.y.sigma = 1 / (state.epsilon.alpha * state.epsilon.beta) + diag([ones(1, N); prediction.G.mu]' * state.be.sigma * [ones(1, N); prediction.G.mu]);
end
