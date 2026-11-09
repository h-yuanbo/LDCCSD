function trace_vec = compute_trace_3_3(u, X, Y, nbrIdx, H, gamma)
% 计算Algorithm 3.3中每个样本的trace值

[d, n] = size(X);           
[k, ~] = size(nbrIdx);     
c = size(Y, 1);
cols = nbrIdx(:);

% 检查Y是否为稀疏矩阵，如果是则转换为满矩阵
if issparse(Y)
    Y = full(Y);
end

% K^o (k×k×n) - 邻居特征的核矩阵
X_tensor = reshape(X(:, cols), d, k, n);
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

% M^o (k×k×n) - 加权核矩阵加正则化
u_neighbors = reshape(u(nbrIdx), k, n);
diag_u_neighbors = zeros(k, k, n);
for i = 1:n
    diag_u_neighbors(:,:,i) = diag(u_neighbors(:,i));
end
M_tensor = pagemtimes(diag_u_neighbors, pagemtimes(K_tensor, diag_u_neighbors)) + gamma * repmat(eye(k), 1, 1, n);

% 计算 trace: tr[(M^o)^{-1} * S^o]
Minv_S = pagemldivide(M_tensor, S_tensor);  % (M^o)^{-1} * S^o
trace_vec = squeeze(sum(sum(Minv_S .* repmat(eye(k), 1, 1, n), 1), 2));  % 对角线元素之和

end
