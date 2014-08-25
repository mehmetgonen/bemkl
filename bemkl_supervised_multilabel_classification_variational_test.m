% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bemkl_supervised_multilabel_classification_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);
    L = length(state.be.mean) - P;

    prediction.G.mean = zeros(P, N, L);
    prediction.G.covariance = zeros(P, N, L);
    for o = 1:L
        for m = 1:P
            prediction.G.mean(m, :, o) = state.A.mean(:, o)' * Km(:, :, m);
            prediction.G.covariance(m, :, o) = state.parameters.sigmag^2 + diag(Km(:, :, m)' * state.A.covariance(:, :, o) * Km(:, :, m));
        end
    end
    
    prediction.F.mean = zeros(L, N);
    prediction.F.covariance = zeros(L, N);
    for o = 1:L
        prediction.F.mean(o, :) = [ones(1, N); prediction.G.mean(:, :, o)]' * state.be.mean([o, L + 1:L + P]);
        prediction.F.covariance(o, :) = 1 + diag([ones(1, N); prediction.G.mean(:, :, o)]' * state.be.covariance([o, L + 1:L + P], [o, L + 1:L + P]) * [ones(1, N); prediction.G.mean(:, :, o)]);
    end    

    pos = 1 - normcdf((+state.parameters.margin - prediction.F.mean) ./ prediction.F.covariance);
    neg = normcdf((-state.parameters.margin - prediction.F.mean) ./ prediction.F.covariance);
    prediction.P = pos ./ (pos + neg);
end