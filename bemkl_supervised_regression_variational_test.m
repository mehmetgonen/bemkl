% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bemkl_supervised_regression_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);

    prediction.G.mean = zeros(P, N);
    prediction.G.covariance = zeros(P, N);
    for m = 1:P
        prediction.G.mean(m, :) = state.a.mean' * Km(:, :, m);
        prediction.G.covariance(m, :) = 1 / (state.upsilon.shape * state.upsilon.scale) + diag(Km(:, :, m)' * state.a.covariance * Km(:, :, m));
    end
    
    prediction.y.mean = [ones(1, N); prediction.G.mean]' * state.be.mean;
    prediction.y.covariance = 1 / (state.epsilon.shape * state.epsilon.scale) + diag([ones(1, N); prediction.G.mean]' * state.be.covariance * [ones(1, N); prediction.G.mean]);
end
