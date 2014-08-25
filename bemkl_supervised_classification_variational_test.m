% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bemkl_supervised_classification_variational_test(Km, state)
    N = size(Km, 2);
    P = size(Km, 3);

    prediction.G.mean = zeros(P, N);
    prediction.G.covariance = zeros(P, N);
    for m = 1:P
        prediction.G.mean(m, :) = state.a.mean' * Km(:, :, m);
        prediction.G.covariance(m, :) = state.parameters.sigmag^2 + diag(Km(:, :, m)' * state.a.covariance * Km(:, :, m));
    end

    prediction.f.mean = [ones(1, N); prediction.G.mean]' * state.be.mean;
    prediction.f.covariance = 1 + diag([ones(1, N); prediction.G.mean]' * state.be.covariance * [ones(1, N); prediction.G.mean]);

    pos = 1 - normcdf((+state.parameters.margin - prediction.f.mean) ./ prediction.f.covariance);
    neg = normcdf((-state.parameters.margin - prediction.f.mean) ./ prediction.f.covariance);
    prediction.p = pos ./ (pos + neg);
end
