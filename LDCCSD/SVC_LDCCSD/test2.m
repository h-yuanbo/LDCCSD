%% test_grad_check.m
% 随机合成一份小数据，调用 obj_grad_u 做梯度校验

clear; clc; rng(42);

% ----- 合成数据尺寸 -----
d = 5;         % 特征维度
c = 4;         % 标签/辅助矩阵行数
n = 30;        % 样本数
k = 6;         % 每个样本的邻居数 (建议 k<=n 且适度)

% ----- 合成数据 -----
X = randn(d,n);
Y = randn(c,n);
G = randn(c,n);

% 构造邻居索引 nbrIdx (这里用随机邻居，实际中请用 kNN)
nbrIdx = zeros(k,n);
for o = 1:n
    idx = randperm(n,k);
    nbrIdx(:,o) = idx(:);
end

% 居中矩阵 H（常用 H = I - (1/k)11^T）
H = eye(k) - ones(k)/k;

% 超参数
gamma = 1e-3;
tau   = 0.1;

% 初始点 u（落在 (0,1) 内，避免边界影响）
u0 = 0.2 + 0.6*rand(n,1);

% 调用解析目标与梯度
[f, g] = obj_grad_u2(u0, X, Y, G, nbrIdx, H, gamma, tau);
fprintf('f(u0) = %.6f,  ||g|| = %.6e\n', f, norm(g));

% ============ 1) 坐标方向中心差分 ===============
eps_fd    = 1e-6;     % 差分步长
max_dims  = min(12,n);% 为了速度，抽样校验的维度数
check_idx = randperm(n, max_dims);
g_num     = zeros(max_dims,1);

for t = 1:max_dims
    i = check_idx(t);
    e_i = zeros(n,1); e_i(i) = 1;

    fp = obj_only(u0 + eps_fd*e_i, X, Y, G, nbrIdx, H, gamma, tau);
    fm = obj_only(u0 - eps_fd*e_i, X, Y, G, nbrIdx, H, gamma, tau);
    g_num(t) = (fp - fm)/(2*eps_fd);
end

g_sel = g(check_idx);
abs_err = abs(g_sel - g_num);
rel_err = abs_err ./ max(1, abs(g_sel) + abs(g_num)); % 元素级相对误差（保守式）

fprintf('\n[坐标方向中心差分]\n');
fprintf('  max |g - g_num|     = %.3e\n', max(abs_err));
fprintf('  mean |g - g_num|    = %.3e\n', mean(abs_err));
fprintf('  max rel error       = %.3e\n', max(rel_err));
fprintf('  mean rel error      = %.3e\n', mean(rel_err));

% ============ 2) 随机方向导数测试 ===============
m_dirs = 5;                   % 测试的随机方向个数
dir_err = zeros(m_dirs,1);
for t = 1:m_dirs
    v = randn(n,1); v = v / norm(v);
    dir_anal = v' * g;

    fp = obj_only(u0 + eps_fd*v, X, Y, G, nbrIdx, H, gamma, tau);
    fm = obj_only(u0 - eps_fd*v, X, Y, G, nbrIdx, H, gamma, tau);
    dir_num = (fp - fm)/(2*eps_fd);

    dir_err(t) = abs(dir_anal - dir_num) / max(1, abs(dir_anal) + abs(dir_num));
end

fprintf('\n[随机方向导数]\n');
fprintf('  max rel error       = %.3e\n', max(dir_err));
fprintf('  mean rel error      = %.3e\n', mean(dir_err));

% ============ 通过/失败简单判据（可按需调整阈值） ============
tol = 1e-4;  % 对中心差分相对误差的经验阈值
ok = (max(rel_err) < tol) && (max(dir_err) < tol);
fprintf('\nGradient check %s (tol = %.1e)\n', ternary(ok,'PASSED','FAILED'), tol);

% ---------- 辅助：只算 f 的包装 ----------
function fval = obj_only(u, X, Y, G, nbrIdx, H, gamma, tau)
    [fval, ~] = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
