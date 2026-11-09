function [f, g] = obj_grad_u_3_3_f1(u, X, Y, nbrIdx, H, gamma, beta)
% 带 logistic 先验负熵正则的目标与梯度 (a=2)

[d, n] = size(X);           
[k, ~] = size(nbrIdx);     
c = size(Y, 1);
cols = nbrIdx(:);

% --- K^o (k×k×n) ---
X_tensor = reshape(full(X(:, cols)), d, k, n);
K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);

% --- H^o ---
H_tensor = repmat(H, 1, 1, n);

% --- Y^o: k×c×n ---
Y_kcn = permute(reshape(Y(:, cols), c, k, n), [2 1 3]);  

% (Y^{oT}Y^o)^{-1/2}
class_counts = sum(Y_kcn, 1);                  
inv_sqrt_counts = 1 ./ (sqrt(class_counts) + eps);  

% Y^o * (Y^{oT}Y^o)^{-1/2}
Y_kcn_norm = Y_kcn .* repmat(inv_sqrt_counts, [k, 1, 1]);  % k×c×n

% B^o = H^o * normed(Y^o)
B_tensor = pagemtimes(H_tensor, Y_kcn_norm);  % k×c×n

% S^o
S_tensor = pagemtimes(B_tensor, permute(B_tensor, [2 1 3]));  % k×k×n

% --- M^o (带 u^o) ---
u_neighbors = reshape(u(nbrIdx), k, n);
diag_u_neighbors = zeros(k, k, n);
for i=1:n
    diag_u_neighbors(:,:,i) = diag(u_neighbors(:,i));
end
M_tensor = pagemtimes(diag_u_neighbors, pagemtimes(K_tensor, diag_u_neighbors)) ...
           + gamma * repmat(eye(k),1,1,n);

% --- trace_vec ---
Minv_S = pagemldivide(M_tensor, S_tensor);  
trace_vec = squeeze(sum(sum(Minv_S .* repmat(eye(k), 1, 1, n), 1), 2));  

% ============ 目标函数部分 =============
% 负熵项 (a=2) => 系数 1/2
epsu = 1e-12;
u_safe = min(max(u, epsu), 1 - epsu);
entropy_term = 0.5 * (u_safe .* log(u_safe) + (1 - u_safe) .* log(1 - u_safe));

f = sum(u .* trace_vec) - beta * sum(u) + sum(entropy_term);

% ============ 梯度计算部分 =============
% MinvS_Minv
Minv_tensor = zeros(k, k, n);
for i = 1:n
    Minv_tensor(:,:,i) = inv(M_tensor(:,:,i));
end
MinvS_Minv = pagemtimes(Minv_tensor, pagemtimes(S_tensor, Minv_tensor));

% Hadamard 乘积: K ∘ (Minv*S*Minv)
Had_tensor = K_tensor .* MinvS_Minv;

% ∂b_o/∂u^o = -2 * (Had ∘ u^o)
db_duo_tensor = pagemtimes(Had_tensor, reshape(u_neighbors, k, 1, n));  
db_duo = -2 * reshape(db_duo_tensor, k, n);

% 初始化梯度: g_i = a_i - beta
g = trace_vec - beta;

% 添加邻居耦合项
rows = nbrIdx(:);  
center_u_rep = repmat(u', k, 1);  
vals = reshape(db_duo, [], 1) .* center_u_rep(:);  
g = g + accumarray(rows, vals, [n, 1], @sum, 0);

% 加上负熵正则梯度
grad_entropy = 0.5 * log(u_safe ./ (1 - u_safe));
g = g + grad_entropy;

end