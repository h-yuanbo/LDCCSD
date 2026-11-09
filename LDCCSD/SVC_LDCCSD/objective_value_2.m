function obj = objective_value_2(u, X, Y, G, nbrIdx, H, gamma, tau)



% Fully vectorized (no per-o loop)
% f(u) = sum_o [ u_o * b_o(u^o) ] - tau * sum_i u_i
% b_o  = tr( Gtilde^o * (M^o)^{-1} * Ytilde^o )
% grad: g_i = b_i - tau + sum_{o in O_i} u_o * (∂b_o/∂u_i),
%       ∂b_o/∂u^o = - (K^o ∘ S^o) u^o,
%       S^o = (M^o)^{-1} (Ytilde^o*Gtilde^o + (Ytilde^o*Gtilde^o)^T) (M^o)^{-1}.

% -------- sizes --------
[d, n] = size(X);           
[k, n2] = size(nbrIdx);     %#ok<ASGLU>
c = size(G,1);
cols = nbrIdx(:);           % (k*n)×1

% -------- K^o (k×k×n) --------
X_tensor = reshape(X(:, cols), d, k, n);                    % d×k×n
K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor); % k×k×n

% -------- Y~^o (k×c×n), G~^o (c×k×n) --------
H_tensor = repmat(H, 1, 1, n);                              % k×k×n

Y_tensor = reshape(Y(:, cols), c, k, n);                    % c×k×n
col_norms = sqrt(sum(Y_tensor.^2, 1)) + eps;                % 1×k×n
Y_norm = Y_tensor ./ col_norms;                             % c×k×n
Ytilde = pagemtimes(H_tensor, permute(Y_norm, [2 1 3]));    % k×c×n

G_tensor = reshape(G(:, cols), c, k, n);                    % c×k×n
G_kc = permute(G_tensor,[2 1 3]);                           % k×c×n
Ghat = permute(pagemtimes(H_tensor, G_kc), [2 1 3]);        % c×k×n

% -------- M^o (k×k×n) --------
u_neighbors = reshape(u(nbrIdx), k, n);                     % k×n
U_outer = pagemtimes( ...
    permute(u_neighbors,[1 3 2]), ...                       % k×1×n
    permute(u_neighbors,[3 1 2]) ...                        % 1×k×n
);                                                          % k×k×n

M_tensor = U_outer .* K_tensor + gamma * repmat(eye(k),1,1,n); % k×k×n

% -------- (M^o)^{-1} by pageinv --------
Minv_tensor = pageinv(M_tensor);                            % k×k×n

% ==================== objective ====================
% A^o = (M^o)^{-1} * Y~^o      (k×c×n)
A_tensor = pagemtimes(Minv_tensor, Ytilde);                 % k×c×n
% F^o = G~^o * A^o             (c×c×n)
F_tensor = pagemtimes(Ghat, A_tensor);                      % c×c×n
% trace per page
I_c_pages = repmat(eye(c), 1, 1, n);
b_vec = squeeze(sum(sum(F_tensor .* I_c_pages, 1), 2));     % n×1

obj = sum(u .* b_vec) - tau * sum(u);
end