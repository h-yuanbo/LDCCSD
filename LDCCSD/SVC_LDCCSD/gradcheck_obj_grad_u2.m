function ok = gradcheck_obj_grad_u2()
%GRADCHECK_OBJ_GRAD_U2  对 obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau) 做梯度校验
% - 坐标方向中心差分
% - 随机方向导数
%
% 返回:
%   ok : 通过(true)/未通过(false)

%% --------- 可调参数 ----------
d = 5; n = 24; c = 4; k = 6;
gamma = 1e-2;          % 适度正则，避免 M 近奇异
tau   = 0.1;
eps_fd = 1e-6;         % 数值差分步长（1e-6~1e-7 之间通常较稳）
num_coords_check = min(12,n);
num_dirs_check   = 6;
tol = 1e-4;            % 通过阈值（可放宽到 1e-3 先验收）

%% --------- 造数据 ----------
rng(0);
X = randn(d,n);
Y = randn(c,n);
G = randn(c,n);
H = eye(k) - ones(k)/k;     % 居中矩阵（推荐）
% 随机邻居（演示用；实际建议用 KNN）
nbrIdx = zeros(k,n);
for o = 1:n
    nbrIdx(:,o) = randperm(n,k);
end
% 初始点（远离边界）
u0 = 0.2 + 0.6*rand(n,1);

%% --------- 解析目标与梯度 ----------
[f0, g0] = obj_grad_u2(u0, X, Y, G, nbrIdx, H, gamma, tau);
fprintf('f(u0) = %.6f,  ||g|| = %.3e\n', f0, norm(g0));

%% --------- 1) 坐标方向中心差分 ----------
coord_sel = randperm(n, num_coords_check);
g_num = zeros(num_coords_check,1);
for t = 1:num_coords_check
    i = coord_sel(t);
    e = zeros(n,1); e(i) = 1;
    fp = obj_only2(u0 + eps_fd*e, X, Y, G, nbrIdx, H, gamma, tau);
    fm = obj_only2(u0 - eps_fd*e, X, Y, G, nbrIdx, H, gamma, tau);
    g_num(t) = (fp - fm) / (2*eps_fd);
end
g_sel   = g0(coord_sel);
abs_err = abs(g_sel - g_num);
rel_err = abs_err ./ max(1, abs(g_sel) + abs(g_num));

fprintf('\n[坐标方向中心差分]\n');
fprintf('  max |g - g_num|  = %.3e\n', max(abs_err));
fprintf('  mean|g - g_num|  = %.3e\n', mean(abs_err));
fprintf('  max rel error    = %.3e\n', max(rel_err));
fprintf('  mean rel error   = %.3e\n', mean(rel_err));

% 可视对比（前 10 个或更少）
mshow = min(10, num_coords_check);
T = table(g_sel(1:mshow), g_num(1:mshow), abs_err(1:mshow), rel_err(1:mshow), ...
          'VariableNames', {'g_analytic','g_numeric','abs_err','rel_err'});
disp('样例对比（解析 vs 数值）：'); disp(T);

%% --------- 2) 随机方向导数 ----------
dir_rel = zeros(num_dirs_check,1);
for t = 1:num_dirs_check
    v = randn(n,1); v = v / norm(v);
    fp = obj_only2(u0 + eps_fd*v, X, Y, G, nbrIdx, H, gamma, tau);
    fm = obj_only2(u0 - eps_fd*v, X, Y, G, nbrIdx, H, gamma, tau);
    dir_num = (fp - fm) / (2*eps_fd);   % 数值方向导数
    dir_ana = v' * g0;                  % 解析方向导数
    dir_rel(t) = abs(dir_ana - dir_num) / max(1, abs(dir_ana) + abs(dir_num));
end
fprintf('\n[随机方向导数]\n');
fprintf('  max rel error    = %.3e\n', max(dir_rel));
fprintf('  mean rel error   = %.3e\n', mean(dir_rel));

%% --------- 通过/失败判据 ----------
ok = (max(rel_err) < tol) && (max(dir_rel) < tol);
fprintf('\nGradient check %s (tol = %.1e)\n', ternary(ok,'PASSED','FAILED'), tol);

end

% ======= 辅助函数 =======
function fval = obj_only2(u, X, Y, G, nbrIdx, H, gamma, tau)
    fval = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
