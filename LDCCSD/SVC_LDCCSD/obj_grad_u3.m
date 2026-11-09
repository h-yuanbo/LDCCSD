function [f, g] = obj_grad_u3(u, X, Y, G, nbrIdx, H, gamma, tau)
%---------------------------------------------------------
% u    : n×1 向量（优化变量）
% X    : d×n 特征矩阵
% Y    : c×n 标签矩阵
% G    : c×n 辅助矩阵
% nbrIdx: k×n 邻居索引
% H    : k×k 居中矩阵
% gamma: 正则化参数 (>0)
% tau  : 超参数 (用于梯度偏移)
%
% f    : 目标函数标量（与 eq:fi 结构一致）
% g    : 梯度 (n×1)，与给定偏导公式一致
%---------------------------------------------------------

[d, n] = size(X); 
[k, ~] = size(nbrIdx);

% ---------- 构造张量 ----------
cols = nbrIdx(:);
X_neighbors = X(:, cols);
X_tensor = reshape(X_neighbors, d, k, n);                         % d×k×n

% K^o = X_N^T X_N
K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);       % k×k×n

% H 张量
H_tensor = repmat(H, 1, 1, n);                                     % k×k×n

% --- Ghat: (c×k×n)
c = size(G, 1);
G_neighbors = G(:, cols);   % c × (k*n)
G_tensor = reshape(G_neighbors, c, k, n);

G_kc = permute(G_tensor, [2 1 3]);% k×c×n
% H 左乘：k×k×n * k×c×n -> k×c×n
HG = pagemtimes(H_tensor, G_kc);
% 转回 c×k×n，得到 Ghat
Ghat = permute(HG, [2 1 3]);   % c×k×n

%Ghat = permute(G_kc, [2 1 3]);                           % c×k×n = (H G^T)^T
% --- Ytilde: (k×c×n)  先列归一化再居中
Y_neighbors = Y(:, cols);                                          % c×(k*n)
Y_tensor = reshape(Y_neighbors, c, k, n);                          % c×k×n

col_norms = sqrt(sum(Y_tensor.^2, 1));                             % 1×k×n
col_norms = col_norms + 1e-12 * max(col_norms, 1);                 % 稳定的“自适应 eps”
Y_norm = Y_tensor ./ col_norms;                                    % c×k×n
Y_norm_kc = permute(Y_norm, [2 1 3]);                              % k×c×n
Ytilde = pagemtimes(H_tensor, Y_norm_kc);                          % k×c×n

% --- u^o 及 M^o
u_neighbors = reshape(u(nbrIdx), k, n);                            % k×n
D_u = permute(u_neighbors, [1 3 2]);                               % k×1×n
U_diag = pagemtimes(D_u, permute(D_u,[2 1 3]));                    % k×k×n

M_tensor = U_diag .* K_tensor;                % diag(u^o)K^o diag(u^o)
I_tensor = repmat(eye(k),1,1,n);
M_tensor = M_tensor + gamma*I_tensor;         % k×k×n
Minv_tensor = pageinv(M_tensor);



     % 批量逆 (需要 R2022b+)
% 第一步： Minv * Ytilde
 tmp = pagemtimes(Minv_tensor, Ytilde);   % k×c×n

% Ghat * tmp
F_tensor = pagemtimes(Ghat, tmp);        % c×c×n   
       % k×k×n (SPD)
trace_vec = squeeze(sum(diag(F_tensor), 1)); % size(trace_vec) = [1×n] 或 [n×1]


% ---------- f: 与 eq:fi 的结构一致 ----------
% f(u) = sum_i (u_i a_i - tau u_i) + sum_o u_o * b_o(u^o)
% 注意：在本设定下 b_o(u^o)=a_o，因而数值上 f = 2*a_vec.'*u - tau*sum(u)
f = 2*(a_vec.'*u) - tau*sum(u);

% ---------- 梯度 g ----------
% S^o = M^{-1} * (Y~ G~ + G~^T Y~^T) * M^{-1}   —— 与公式一致的“夹心”形式
T1 = pagemtimes(Ytilde, Ghat);                                     % k×k×n  = Y~ * G~
T2 = pagemtimes(G_kc, permute(Ytilde,[2 1 3]));   % ✅ k×k×n = (k×c×n) * (c×k×n)

T  = T1 + T2;                                                      % k×k×n

% 这里需要右侧再乘一次 M^{-1}。MATLAB 无 paged 右除，
% 为严格对应公式，采用一次 pageinv 来得到 M^{-1}（只在此处使用）。
Minv_tensor = pageinv(M_tensor);                                   % k×k×n
S = pagemtimes(pagemtimes(Minv_tensor, T), Minv_tensor);           % k×k×n

% Hadamard: (K^o ∘ S^o)
Had = S .* K_tensor;                                               % k×k×n

% 计算 - (K^o ∘ S^o) u^o 的 paged mat-vec（向量化）
u_pages   = permute(u_neighbors, [1 3 2]);                         % k×1×n
db_duo_pg = -pagemtimes(Had, u_pages);                             % k×1×n
db_duo    = squeeze(db_duo_pg);                                    % k×n

% g_i = a_i - tau + sum_{o∈O_i} u_o * (∂b_o/∂u_i)
g = a_vec - tau*ones(n,1,'like',u);

% 把每页的 k 维分量按 nbrIdx 散射回全局 i，并乘上 u_o
rows = nbrIdx(:);                                                  % 全局索引 i
vals = reshape(db_duo .* repmat(u.', k, 1), [], 1);                % 乘以 u_o
g = g + cast(accumarray(double(nbrIdx(:)), vals, [n,1], @sum, 0), 'like', u);


end
