function ok = test_obj_grad_u2_all()
%TEST_OBJ_GRAD_U2_ALL
% 比较你的 obj_grad_u2 与“手算（朴素实现）”以及数值梯度的一致性。
% 通过则返回 true，并打印关键误差指标。

%% ===== 1) 造一份可复现实验数据 =====
rng(0);
d = 6; n = 25; c = 4; k = 5;

% 稳健的默认参数
gamma = 1e-2;
tau   = 0.1;
H = eye(k) - ones(k)/k;   % 居中矩阵

X = randn(d,n);
Y = randn(c,n);
G = randn(c,n);

% 邻居索引（演示随机；实际请换为 KNN 结果）
nbrIdx = zeros(k,n);
for o = 1:n
    nbrIdx(:,o) = randperm(n,k);
end

% 初始点（远离边界）
u = 0.2 + 0.6*rand(n,1);

%% ===== 2) f 与 朴素 f-only 对比（手算公式） =====
[f_fast, g_fast] = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau);
f_naive = f_only_naive(u, X, Y, G, nbrIdx, H, gamma, tau);

diff_f = abs(f_fast - f_naive);
fprintf('f_fast = %.12f | f_naive = %.12f | |diff| = %.3e\n', f_fast, f_naive, diff_f);

%% ===== 3) 数值梯度（中心差分） vs 解析梯度 =====
eps_fd  = 1e-6;
m_check = min(12,n);
idx     = randperm(n, m_check);
g_num   = zeros(m_check,1);

for t = 1:m_check
    i = idx(t);
    e = zeros(n,1); e(i) = 1;
    fp = obj_only(u + eps_fd*e, X, Y, G, nbrIdx, H, gamma, tau);
    fm = obj_only(u - eps_fd*e, X, Y, G, nbrIdx, H, gamma, tau);
    g_num(t) = (fp - fm) / (2*eps_fd);
end

abs_err = abs(g_fast(idx) - g_num);
rel_err = abs_err ./ max(1, abs(g_fast(idx)) + abs(g_num));

fprintf('\n[坐标中心差分]\n');
fprintf('  max |g-g_num|  = %.3e\n', max(abs_err));
fprintf('  mean|g-g_num|  = %.3e\n', mean(abs_err));
fprintf('  max rel error  = %.3e\n', max(rel_err));
fprintf('  mean rel error = %.3e\n', mean(rel_err));

%% ===== 4) 随机方向导数一致性 =====
T = 6;
rel_dir = zeros(T,1);
for t = 1:T
    v = randn(n,1); v = v/norm(v);
    fp = obj_only(u + eps_fd*v, X, Y, G, nbrIdx, H, gamma, tau);
    fm = obj_only(u - eps_fd*v, X, Y, G, nbrIdx, H, gamma, tau);
    dir_num = (fp - fm) / (2*eps_fd);
    dir_ana = v' * g_fast;
    rel_dir(t) = abs(dir_ana - dir_num) / max(1, abs(dir_ana) + abs(dir_num));
end
fprintf('\n[随机方向导数]\n');
fprintf('  max rel error  = %.3e\n', max(rel_dir));
fprintf('  mean rel error = %.3e\n', mean(rel_dir));

%% ===== 5) 判定 =====
tol_f   = 1e-8;    % f 对比阈值
tol_grad= 1e-4;    % 梯度相对误差阈值
ok = (diff_f < tol_f) && (max(rel_err) < tol_grad) && (max(rel_dir) < tol_grad);
fprintf('\nSummary: f_diff < %.1e ? %d | grad_checks < %.1e ? %d  -->  %s\n', ...
    tol_f, diff_f < tol_f, tol_grad, (max(rel_err) < tol_grad) && (max(rel_dir) < tol_grad), ...
    ternary(ok,'PASSED','FAILED'));
end

%% ======== 只算 f 的包装（直接调你的函数） ========
function fval = obj_only(u, X, Y, G, nbrIdx, H, gamma, tau)
    fval = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau);
end

%% ======== 朴素 f-only（手算定义，逐页）========
function fval = f_only_naive(u, X, Y, G, nbrIdx, H, gamma, tau)
% f(u) = sum_o [ u_o * b_o(u^o) ] - tau * sum_i u_i
% b_o = tr( G~^o * (M^o)^{-1} * Y~^o ),
% M^o = diag(u^o) * K^o * diag(u^o) + gamma*I_k,
% K^o = (X^o)' * X^o,
% Y~^o = H * (Y^o ./ ||列||),  G~^o = (H * (G^o)')'
[~, n] = size(X);
k = size(nbrIdx,1);
fsum = 0;
for o = 1:n
    idx = nbrIdx(:,o);
    Xo = X(:,idx);              % d×k
    Ko = Xo' * Xo;              % k×k
    uo = u(idx);                % k×1
    Mo = diag(uo) * Ko * diag(uo) + gamma * eye(k);   % k×k

    Yo = Y(:,idx);                             % c×k
    cols_norm = sqrt(sum(Yo.^2,1)) + eps;      % 1×k
    Ynorm = Yo ./ cols_norm;                   % c×k
    Ytilde = H * (Ynorm.');                    % k×c  ← 注意转置

    Go = G(:,idx);                             % c×k
    Gtilde = (H * Go')';                       % c×k

    MinvY = Mo \ Ytilde;                       % k×c
    Fo = Gtilde * MinvY;                       % c×c
    bo = trace(Fo);
    fsum = fsum + u(o) * bo;
end
fval = fsum - tau * sum(u);
end

%% ======== 小工具 ========
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
