function [f, g] = obj_grad_u_3_3(u, X, Y, nbrIdx, H, gamma, tau)

[d, n] = size(X);           
[k, ~] = size(nbrIdx);     
c = size(Y, 1);
cols = nbrIdx(:);



% K^o (k×k×n)
%X_tensor = reshape(X(:, cols), d, k, n);
X_tensor = reshape(full(X(:, cols)), d, k, n);

K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);

% 计算 B^o = H^o * Y^o * (Y^{oT}Y^o)^{-1/2}
H_tensor = repmat(H, 1, 1, n);
Y_kcn = permute(reshape(Y(:, cols), c, k, n), [2 1 3]);  % k×c×n

% 计算 (Y^{oT}Y^o)^{-1/2}
class_counts = sum(Y_kcn, 1);  % 1×c×n
inv_sqrt_counts = 1 ./ (sqrt(class_counts) + eps);  % 1×c×n

% Y^o * (Y^{oT}Y^o)^{-1/2}
Y_kcn_norm = Y_kcn .* repmat(inv_sqrt_counts, [k, 1, 1]);  % k×c×n

% B^o = H^o * Y^o * (Y^{oT}Y^o)^{-1/2}
B_tensor = pagemtimes(H_tensor, Y_kcn_norm);  % k×c×n

% S^o = B^o * (B^o)^T
S_tensor = pagemtimes(B_tensor, permute(B_tensor, [2 1 3]));  % k×k×n

% M^o (k×k×n)
u_neighbors = reshape(u(nbrIdx), k, n);
diag_u_neighbors = zeros(k, k, n);
for i=1:n
    diag_u_neighbors(:,:,i) = diag(u_neighbors(:,i));
end
M_tensor = pagemtimes(diag_u_neighbors, pagemtimes(K_tensor, diag_u_neighbors)) + gamma * repmat(eye(k),1,1,n);

% 目标函数: a_i = tr[(M^i)^{-1} * S^i]
Minv_S = pagemldivide(M_tensor, S_tensor);  % (M^o)^{-1} * S^o
trace_vec = squeeze(sum(sum(Minv_S .* repmat(eye(k), 1, 1, n), 1), 2));  % 对角线元素之和
f = sum(u .* trace_vec) - tau * sum(u);

% 梯度计算
% 计算 (M^o)^{-1} * S^o * (M^o)^{-1}
Minv_tensor = zeros(k, k, n);
for i = 1:n
    Minv_tensor(:,:,i) = pinv(M_tensor(:,:,i));
end
MinvS_Minv = pagemtimes(Minv_tensor, pagemtimes(S_tensor, Minv_tensor));  % k×k×n

% Hadamard乘积: K^o ○ [(M^o)^{-1} * S^o * (M^o)^{-1}]
Had_tensor = K_tensor .* MinvS_Minv;  % k×k×n

% ∂b_o/∂u^o = -2 * (K^o ○ [(M^o)^{-1} * S^o * (M^o)^{-1}]) * u^o
db_duo_tensor = pagemtimes(Had_tensor, reshape(u_neighbors, k, 1, n));  % k×1×n
db_duo = -2 * reshape(db_duo_tensor, k, n);  % k×n，注意系数-2

% 初始化梯度: g_i = a_i - tau
g = trace_vec - tau;

% 添加耦合项: sum_{o in O_i} u_o * (∂b_o/∂u_i)
rows = nbrIdx(:);  % (k*n)×1
center_u_rep = repmat(u', k, 1);  % k×n
vals = reshape(db_duo, [], 1) .* center_u_rep(:);  % (k*n)×1

g = g + accumarray(rows, vals, [n, 1], @sum, 0);

end
