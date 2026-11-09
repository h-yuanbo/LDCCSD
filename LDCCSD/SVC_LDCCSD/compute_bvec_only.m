function b_vec = compute_bvec_only(u, X, G, nbrIdx, H, gamma)
% Fully vectorized b_vec computation (no per-o loop)
% Inputs:
%   u      : n×1
%   X      : d×n
%   Y, G   : c×n
%   nbrIdx : k×n (each column lists k neighbors of center o)
%            if provided as n×k, will be transposed internally
%   H      : k×k
%   gamma  : scalar > 0
% Output:
%   b_vec  : n×1, with b_o = tr( Gtilde^o * (M^o)^{-1} * Ytilde^o )

    % ---------- shapes ----------
    [d, n] = size(X); 
    c = size(G,1);

    % normalize nbrIdx to k×n
    if size(nbrIdx,2) ~= n && size(nbrIdx,1) == n
        nbrIdx = nbrIdx.';           % n×k -> k×n
    elseif size(nbrIdx,2) ~= n
        error('nbrIdx must be k×n or n×k.');
    end
    k = size(nbrIdx,1);

    % ---------- gather neighbors & build tensors ----------
    cols      = nbrIdx(:);                                    % (k*n)×1
    X_tensor  = reshape(X(:, cols), d, k, n);                 % d×k×n
    K_tensor  = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);  % k×k×n

    H_tensor  = repmat(H, 1, 1, n);                           % k×k×n

                 % c×k×n
    
    
    G_tensor  = reshape(G(:, cols), c, k, n);                 % c×k×n
    G_kc      = permute(G_tensor,[2 1 3]);                    % k×c×n
    Ghat      = permute(pagemtimes(H_tensor, G_kc), [2 1 3]); % c×k×n
    %Ytilde    = pagemtimes(H_tensor, permute(Y_norm, [2 1 3]));   % k×c×n
    % ---------- build M^o in batch ----------
    u_neighbors = reshape(u(nbrIdx), k, n);                   % k×n
    U_outer     = pagemtimes( ...
                      permute(u_neighbors,[1 3 2]), ...       % k×1×n
                      permute(u_neighbors,[3 1 2]) ...        % 1×k×n
                   );                                         % k×k×n
    M_tensor    = U_outer .* K_tensor + gamma * repmat(eye(k),1,1,n); % k×k×n
    Ghat2     = permute(pagemtimes(H_tensor, G_kc), [1 2 3]); % c×k×n
    % ---------- inverse & objective in batch ----------
    A_tensor = pagemldivide(M_tensor, Ghat2);
    F_tensor    = pagemtimes(Ghat, A_tensor);                 % c×c×n

    % pagewise trace
    I_c_pages   = repmat(eye(c), 1, 1, n);                    % c×c×n
    b_vec       = squeeze(sum(sum(F_tensor .* I_c_pages, 1), 2)); % n×1
    %fprintf('      gradient b_vec: min=%.6f, max=%.6f, mean=%.6f\n', ...
        %min(b_vec), max(b_vec), mean(b_vec));

    
end
