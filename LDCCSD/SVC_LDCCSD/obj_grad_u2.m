function [f, g] = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau)
% Fully vectorized (no per-o loop)
% f(u) = sum_o [ u_o * b_o(u^o) ] - tau * sum_i u_i
% b_o  = tr( Gtilde^o * (M^o)^{-1} * Ytilde^o )
% grad: g_i = b_i - tau + sum_{o in O_i} u_o * (∂b_o/∂u_i),
%       ∂b_o/∂u^o = - (K^o ∘ S^o) u^o,
%       S^o = (M^o)^{-1} (Ytilde^o*Gtilde^o + (Ytilde^o*Gtilde^o)^T) (M^o)^{-1}.

% -------- sizes --------
[d, n] = size(X);           
[k, ~] = size(nbrIdx);     
c = size(G,1);
cols = nbrIdx(:);           % (k*n)×1

% -------- K^o (k×k×n) --------
X_tensor = reshape(X(:, cols), d, k, n);                    % d×k×n
K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor); % k×k×n

% -------- Y~^o (k×c×n), G~^o (c×k×n) --------
H_tensor = repmat(H, 1, 1, n);                              % k×k×n

% 1) 取出邻域的 one-hot：k×c×n
Y_kcn = permute(reshape(Y(:, cols), c, k, n), [2 1 3]);  % k×c×n

% 2) 每个 o 的“按类计数”：1×c×n   （就是 Y^{oT} Y^o 的对角）
class_counts = sum(Y_kcn, 1);                 % 1×c×n
inv_sqrt = 1 ./ sqrt(class_counts + eps);     % 1×c×n

% 3) 右乘 (Y^{oT}Y^o)^{-1/2} 通过广播：对每页、每个类列缩放
Y_kcn_norm = Y_kcn .* repmat(inv_sqrt, [size(Y_kcn,1), 1, 1]); % k×c×n

% 4) 乘以 H：H^o * (Y^o * (Y^{oT}Y^o)^{-1/2})
Ytilde = pagemtimes(H_tensor, Y_kcn_norm);    % k×c×n


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
%Minv_tensor = pageinv(M_tensor);                            % k×k×n
A_tensor = pagemldivide(M_tensor, Ytilde);
% ==================== objective ====================
% A^o = (M^o)^{-1} * Y~^o      (k×c×n)
%A_tensor = pagemtimes(Minv_tensor, Ytilde);                 % k×c×n
% F^o = G~^o * A^o             (c×c×n)
F_tensor = pagemtimes(Ghat, A_tensor);                      % c×c×n
% trace per page
I_c_pages = repmat(eye(c), 1, 1, n);
trace_vec = squeeze(sum(sum(F_tensor .* I_c_pages, 1), 2));     % n×1

f = sum(u .* trace_vec) - tau * sum(u);

% ==================== gradient ====================
% P^o = Y~^o * G~^o            (k×k×n)
P_tensor = pagemtimes(Ytilde, Ghat);       % k×k×n
S0_tensor = P_tensor + permute(P_tensor,[2 1 3]);           % k×k×n, sym
% S^o = Minv * S0 * Minv       (k×k×n)
S_tensor = pagemrdivide(pagemldivide(M_tensor, S0_tensor), M_tensor);

% Hadamard with K^o
Had_tensor = S_tensor .* K_tensor;                          % k×k×n

% db_duo(:,o) = - (Had^o * u^o)
% -> use pagemtimes with (k×k×n) and (k×1×n)
db_duo_tensor = pagemtimes(Had_tensor, reshape(u_neighbors, k, 1, n)); % k×1×n
db_duo = - reshape(db_duo_tensor, k, n);                    % k×n

% g = b - tau + scatter( u_o * (∂b_o/∂u_i) )
g = trace_vec - tau * ones(n,1);

rows = nbrIdx(:);                                           % (k*n)×1
vals = reshape(db_duo, [], 1) .* reshape(repmat(u.', k, 1), [], 1);
g = g + accumarray(rows, vals, [n, 1], @sum, 0);



end
