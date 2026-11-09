function ok = check_f_only_obj_grad_u2()
%CHECK_F_ONLY_OBJ_GRAD_U2
% 验证 obj_grad_u2 返回的 f 是否与“朴素定义实现”的 f 一致；
% 另外做一个方向差分一致性小测（不依赖梯度）。

%% 1) 造一份可复现实验数据
rng(0);
d = 6; n = 25; c = 4; k = 5;

% 推荐的稳健配置
gamma = 1e-2;          % 正则别太小，避免 M 近奇异
tau   = 0.1;           % 任意设一个
H = eye(k) - ones(k)/k; % 常用的居中矩阵

X = randn(d,n);
Y = randn(c,n);
G = randn(c,n);

% 邻居索引（演示用随机；实际请替换为你的 KNN 索引）
nbrIdx = zeros(k,n);
for o = 1:n
    nbrIdx(:,o) = randperm(n,k);
end

%% 2) 在多个随机 u 上比较 f_fast vs f_naive
T = 5;                                % 测试轮数
U = 0.2 + 0.6*rand(n,T);              % 远离边界，数值更稳
diffs = zeros(T,1);

for t = 1:T
    u = U(:,t);

    % (a) 用你的函数算 f（忽略梯度）
    f_fast = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau);

    % (b) 用严格按定义的朴素实现计算 f
    f_naive = f_only_naive(u, X, Y, G, nbrIdx, H, gamma, tau);

    diffs(t) = abs(f_fast - f_naive);
    fprintf('Trial %d: f_fast = %.12f | f_naive = %.12f | |diff| = %.3e\n', ...
            t, f_fast, f_naive, diffs(t));
end

%% 3) 一个方向差分一致性小测（不依赖梯度）
% 仅检测 f 的自洽性：phi(eps) = f(u + eps*v)，看两侧差分是否对称稳定
fprintf('\n[方向差分一致性小测]\n');
u0  = 0.2 + 0.6*rand(n,1);
v   = randn(n,1); v = v/norm(v);
eps = 1e-6;
f_p = obj_grad_u2(u0 + eps*v, X, Y, G, nbrIdx, H, gamma, tau);
f_m = obj_grad_u2(u0 - eps*v, X, Y, G, nbrIdx, H, gamma, tau);
sym_diff = abs((f_p - f_m));              % 对称差分应 ~ O(eps)
fprintf('  |f(u+eps v) - f(u-eps v)| = %.3e  (eps = %.1e)\n', sym_diff, eps);

%% 4) 判定
tol = 1e-8;          % 双精度下一般可到 1e-10~1e-12，这里给宽松阈值
ok = all(diffs < tol);
fprintf('\nMax |f_fast - f_naive| = %.3e, tol = %.1e  -->  %s\n', ...
        max(diffs), tol, ternary(ok,'PASSED','FAILED'));
end

% ----------------- 朴素 f-only（严格按定义，无 3D 批量技巧） -----------------
function fval = f_only_naive(u, X, Y, G, nbrIdx, H, gamma, tau)
    [~, n] = size(X);
    k = size(nbrIdx,1);
    fsum = 0;
    for o = 1:n
        idx = nbrIdx(:,o);
        Xo = X(:,idx);                % d×k
        Ko = Xo' * Xo;                % k×k

        uo = u(idx);                  % k×1
        Mo = diag(uo) * Ko * diag(uo) + gamma * eye(k);    % k×k

        Yo = Y(:,idx);                                      % c×k
        cols_norm = sqrt(sum(Yo.^2,1)) + eps;               % 1×k
        Ynorm = Yo ./ cols_norm;                            % c×k
        Ytilde = H * (Ynorm.');                             % k×c  ← 修正点

        Go = G(:,idx);                                      % c×k
        Gtilde = (H * Go')';                                % c×k

        MinvY = Mo \ Ytilde;                                % k×c
        Fo = Gtilde * MinvY;                                % c×c
        bo = trace(Fo);

        fsum = fsum + u(o) * bo;
    end
    fval = fsum - tau * sum(u);
end


% ----------------- 小工具 -----------------
function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
