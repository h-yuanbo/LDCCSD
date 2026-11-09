function [f, g] = obj_grad_u(u, X, Y, G, nbrIdx, H, gamma, tau)
%---------------------------------------------------------
% u    : n×1 向量（优化变量）
% X    : d×n 特征矩阵
% Y    : c×n 标签矩阵
% G    : c×n 辅助矩阵
% Idx  : k×n 邻居索引
% H    : k×k 居中矩阵
% gamma: 正则化参数
% tau  : 超参数 (用于梯度偏移)
%
% f    : 目标函数标量
% g    : 梯度 (n×1)
%---------------------------------------------------------

[d, n] = size(X);
[k, ~] = size(nbrIdx);

%% ---------- 构造张量 ----------
cols = nbrIdx(:);
X_neighbors = X(:, cols);
X_tensor = reshape(X_neighbors, d, k, n);               % d×k×n

% K^o
K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor); % k×k×n
H_tensor = repmat(H,1,1,n);

% --- Ghat ---
c = size(G, 1);
G_neighbors = G(:, cols);                 % c × (k*n)
G_tensor = reshape(G_neighbors, c, k, n); % c×k×n
G_kc = permute(G_tensor,[2 1 3]);         % k×c×n
HG = pagemtimes(H_tensor, G_kc);          % k×c×n
Ghat = permute(HG, [2 1 3]);              % c×k×n

% --- Ytilde ---
Y_neighbors = Y(:, cols);                 % c × (k*n)
Y_tensor = reshape(Y_neighbors, c, k, n); % c×k×n

col_norms = sqrt(sum(Y_tensor.^2, 1)) + eps;  % 1×k×n
Y_norm = Y_tensor ./ col_norms;                % c×k×n
Y_norm_kc = permute(Y_norm, [2 1 3]);          % k×c×n
Ytilde = pagemtimes(H_tensor, Y_norm_kc);      % k×c×n

% --- u^o 构造 M^o ---
u_neighbors = reshape(u(nbrIdx), k, n);       % k×n
U_diag = pagemtimes( ...
              permute(u_neighbors,[1 3 2]), ... % k×1×n
              permute(u_neighbors,[3 1 2]) ...  % 1×k×n
           );                                  % k×k×n

M_tensor = U_diag .* K_tensor;                % diag(u^o)K^o diag(u^o)
I_tensor = repmat(eye(k),1,1,n);
M_tensor = M_tensor + gamma*I_tensor;         % k×k×n
Minv_tensor = pageinv(M_tensor);



     % 批量逆 (需要 R2022b+)
% 第一步： Minv * Ytilde
 tmp = pagemtimes(Minv_tensor, Ytilde);   % k×c×n

% Ghat * tmp
%F_tensor = pagemtimes(Ghat, tmp);        % c×c×n   

    %% ---------- 目标函数计算 ----------
A = pagemtimes(Minv_tensor, Ytilde);         % k×c×n   = Minv * Ytilde
F_tensor = pagemtimes(Ghat, A);              % c×c×n

% 每页 trace：不要对 3D 直接 diag，这里用与单位阵相乘求和的 trick
I_c = eye(c);
I_c_pages = repmat(I_c, 1, 1, n);            % c×c×n
trace_vec = squeeze(sum(sum(F_tensor .* I_c_pages, 1), 2)); % n×1

% 线性项：a_i u_i - tau u_i
% 这里采用 a_i = trace_vec（与 b_o 同构、页=样本的常见设定）
a_vec = trace_vec;                            % n×1
f = sum(trace_vec) + a_vec.'*u - tau * sum(u);
%% Step 5: 梯度 g
B   = pagemtimes(Minv_tensor, G_kc);                     % k×c×n = Minv * G_kc
S1  = pagemtimes(A,Ghat);       % k×k×n = A * Ghat^T
S2  = pagemtimes(B,permute(Ytilde,[2 1 3]));       % k×k×n = B * Ytilde^T
S   = S1 + S2;
S   = pagemtimes(pagemtimes(Minv_tensor, S), Minv_tensor); % k×k×n

% ∂b_o/∂u^o = - (K^o ∘ S^o) u^o
Had   = S .* K_tensor;                        % k×k×n
db_duo = zeros(k,n);
for t = 1:n
    db_duo(:,t) = - Had(:,:,t) * u_neighbors(:,t);
end

% g_i = a_i - tau + ∑_{o∈O_i} u_o * (∂b_o/∂u_i)
g = a_vec - tau*ones(n,1);

% 把每页的 k 维分量按 nbrIdx 散射回全局 i
rows = nbrIdx(:);                                           % 要累加到的全局索引 i
vals = reshape(db_duo, [], 1) .* reshape(repmat(u.', k, 1), [], 1); % 乘以 u_o
g = g + accumarray(rows, vals, [n, 1], @sum, 0);

end



