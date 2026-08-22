function I_hat = haran_filter(I, K, M, p)
% Haran filter implementation following Haran, J. Chem. Phys. (2004)

L = length(I);

% pad intensity vector
I = I(:);
I = [zeros(max(K),1); I; zeros(max(K),1)];

num_predictors = length(K);
predictors = cell(num_predictors,2);

forward_weights = zeros(L, num_predictors);
reverse_weights = zeros(L, num_predictors);

for ii = 1:num_predictors
    Kii = K(ii);
    predictors{ii,1} = forward_predictor(I, Kii);
    predictors{ii,2} = reverse_predictor(I, Kii);

    % compute weights
    fp = predictors{ii,1}; % forward predictor
    rp = predictors{ii,2}; % reverse predictor
    for jj = M:L-M
        for kk = 0:M-1
            forward_weights(jj,ii) = forward_weights(jj,ii) + (I(jj-kk) - fp(jj-kk));
            reverse_weights(jj,ii) = reverse_weights(jj,ii) + (I(jj-kk) - rp(jj-kk));
        end
    end
end

forward_weights = forward_weights .^ (-p);
reverse_weights = reverse_weights .^ (-p);

% normalize weights
Z = 1 ./ (sum(forward_weights,2) + sum(reverse_weights,2));
forward_weights = Z .* forward_weights;
reverse_weights = Z .* reverse_weights;

I_hat = zeros(L,1);
for ii = 1:num_predictors
    Kii = max(K);
    fp = predictors{ii,1};
    rp = predictors{ii,2};

    fp = fp(Kii:end-Kii-1);
    rp = rp(Kii:end-Kii-1);

    fw = forward_weights(:,ii);
    rw = reverse_weights(:,ii);
    I_hat = I_hat + ((fw.*fp) + (rw.*rp));
end
end

function I_hat = forward_predictor(I, N)
I_hat = zeros(size(I));
for ii = 1+N:length(I)-N
    I_hat(ii) = (1./N).*sum(I(ii-N:ii-1));
end
end

function I_hat = reverse_predictor(I, N)
I_hat = zeros(size(I));
for ii = 1+N:length(I)-N
    I_hat(ii) = (1./N).*sum(I(ii+1:ii+N));
end
end
