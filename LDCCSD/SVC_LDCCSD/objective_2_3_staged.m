function [labels_pred, history] = objective_2_3_staged(X, Y_true, params)
% 外层：逐步增大 tau（课程）；内层：固定 tau 交替优化直到收敛
% 依赖：build_knn_graph, assemble_Lu_for_2_3, solve_Y, compute_trace_values_2_3, my_eval_y

n = size(X,1);
c = params.nCluster;
k = params.k;
H = eye(k) - (1/k)*ones(k);

tau      = params.tau;          % 初始 τ
tau_max  = params.tau_max;
taudelt  = params.taudelt;      % 每个阶段结束后才增长
numStage = 5;     % 外层阶段数（建议 5~8）
innerMax = 20;     % 每阶段内层最大迭代（建议 20~50）
alpha    = 0.2;                 % 软化系数，避免 HY≈0
beta_u   = 0.5;                 % u 的动量/平滑（0.3~0.7）

nbrIdx = build_knn_graph(X, k);

% ===== 初始化 =====
u = zeros(n,1);
L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
[V,D] = eig(full(L_u)); [~,idx] = sort(diag(D),'ascend');
G = real(V(:, idx(2:c+1)));
[labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',1);
Y = full(ind2vec(labels_pred')');

% ===== 历史记录 =====
history = struct();
history.stage_tau = zeros(numStage,1);
history.stage_curves = cell(numStage,1);   % 每阶段内层 OFV 曲线
history.Accuracy = [];

for s = 1:numStage
    % 固定当前 tau，内层收敛
    fvals = zeros(innerMax,1);
    fprev = NaN;  Yprev = Y;  uprev = u;

    for it = 1:innerMax
        % 1) 固定 u 更新 Y（软化）
        L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        [Y_hat, ~] = solve_Y(L_u, Y);
        Y = (1 - alpha)*Y + alpha*Y_hat;

        % 2) 用软 Y 计算 trace（保持可分辨）
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        trace_vec = max(trace_vec, 1e-12);              % 数值稳健

        % 3) 固定 tau 全量更新 u（闭式 + 平滑）
        u_new = max(0, 1 - (trace_vec ./ tau));         % v* = max(0, 1 - loss/tau)
        u = (1 - beta_u)*u + beta_u*u_new;              % 平滑，抑制抖动

        % 4) 目标函数（与你的正则一致）
        fval = sum(u .* trace_vec) + tau * (0.5*(u.'*u) - sum(u));
        fvals(it) = fval;

        % 5) 收敛判据（固定 tau 的内层）
        if it > 1
            relF = abs(fval - fprev) / (abs(fprev) + 1e-8);
            relU = norm(u - uprev) / (norm(uprev) + 1e-8);
            relY = norm(Y - Yprev,'fro') / (norm(Yprev,'fro') + 1e-8);
            if relF < 1e-4 && relU < 1e-3 && relY < 1e-3
                fvals = fvals(1:it);
                break;
            end
        end
        fprev = fval; uprev = u; Yprev = Y;

        % 可选：监控分类指标
        [~, labels_pred] = max(Y,[],2);
        r = my_eval_y(labels_pred, Y_true);
        history.Accuracy(end+1,1) = r.Accuracy;
    end

    % 记录该阶段曲线与 tau
    history.stage_tau(s) = tau;
    history.stage_curves{s} = fvals;

    % 外层提升 tau（进入更难样本）
    if tau < tau_max
        tau = min(tau * taudelt, tau_max);
    end
end

% 最终标签
[~, labels_pred] = max(Y,[],2);
end
