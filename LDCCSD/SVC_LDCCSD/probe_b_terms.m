function b_vec = probe_b_terms(u, X, Y, G, nbrIdx, H, gamma)
    [d, n] = size(X);
    [k, ~] = size(nbrIdx);
    c = size(G,1);
    cols = nbrIdx(:);

    X_tensor = reshape(X(:, cols), d, k, n);
    K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);

    H_tensor = repmat(H, 1, 1, n);
    Y_tensor = reshape(Y(:, cols), c, k, n);
    col_norms = sqrt(sum(Y_tensor.^2, 1)) + eps;
    Y_norm = Y_tensor ./ col_norms;
    Ytilde = pagemtimes(H_tensor, permute(Y_norm, [2 1 3])); % k×c×n

    G_tensor = reshape(G(:, cols), c, k, n);
    G_kc = permute(G_tensor,[2 1 3]);
    Ghat = permute(pagemtimes(H_tensor, G_kc), [2 1 3]);     % c×k×n

    u_neighbors = reshape(u(nbrIdx), k, n);
    U_outer = pagemtimes(permute(u_neighbors,[1 3 2]), permute(u_neighbors,[3 1 2]));
    M_tensor = U_outer .* K_tensor + gamma * repmat(eye(k),1,1,n);

    A_tensor = pagemldivide(M_tensor, Ytilde);         % (M^o)^{-1} Y~
    F_tensor = pagemtimes(Ghat, A_tensor);             % c×c×n
    I_c_pages = repmat(eye(c), 1, 1, n);
    b_vec = squeeze(sum(sum(F_tensor .* I_c_pages, 1), 2));  % 每个 o 的 b_o
end
