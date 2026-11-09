function cache = make_u_cache(X, Y, G, nbrIdx, H, gamma)
[d, n] = size(X);
[k, ~] = size(nbrIdx);
c = size(G,1);

cols = nbrIdx(:);

X_tensor = reshape(X(:, cols), d, k, n);
K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);

H_tensor = repmat(H,1,1,n);

Y_tensor = reshape(Y(:, cols), c, k, n);
col_norms = sqrt(sum(Y_tensor.^2, 1)) + eps;
Y_norm = Y_tensor ./ col_norms;
Ytilde = pagemtimes(H_tensor, permute(Y_norm,[2 1 3]));

G_tensor = reshape(G(:, cols), c, k, n);
G_kc = permute(G_tensor,[2 1 3]);
Ghat = permute(pagemtimes(H_tensor, G_kc), [2 1 3]);

cache = struct('K_tensor',K_tensor,'Ytilde',Ytilde,'Ghat',Ghat, ...
               'G_kc',G_kc,'nbrIdx',nbrIdx,'gamma',gamma);
end
