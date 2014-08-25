% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bemkl_supervised_multioutput_regression_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);
    L = length(state.be.mean) - P;    
    
    prediction.G.mean = zeros(P, N, L);
    prediction.G.covariance = zeros(P, N, L);
    for o = 1:L
        for m = 1:P
            prediction.G.mean(m, :, o) = state.A.mean(:, o)' * Km(:, :, m);
            prediction.G.covariance(m, :, o) = 1 / (state.upsilon.shape(o) * state.upsilon.scale(o)) + diag(Km(:, :, m)' * state.A.covariance(:, :, o) * Km(:, :, m));
        end
    end   
    
    prediction.Y.mean = zeros(L, N);
    prediction.Y.covariance = zeros(L, N);
    for o = 1:L
        prediction.Y.mean(o, :) = [ones(1, N); prediction.G.mean(:, :, o)]' * state.be.mean([o, L + 1:L + P]);
        prediction.Y.covariance(o, :) = 1 / (state.epsilon.shape(o) * state.epsilon.scale(o)) + diag([ones(1, N); prediction.G.mean(:, :, o)]' * state.be.covariance([o, L + 1:L + P], [o, L + 1:L + P]) * [ones(1, N); prediction.G.mean(:, :, o)]);
    end
end
