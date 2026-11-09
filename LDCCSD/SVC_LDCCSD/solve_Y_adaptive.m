function [F, obj] = solve_Y_adaptive(K, F, maxIter, base_lambda)
% Adaptive batch-update solver on Y
% K: 核矩阵
% F: 初始指示矩阵 n×c
% maxIter: 最大迭代轮数
% base_lambda: 基础平衡惩罚强度（建议 0.5 ~ 1）

if nargin < 3, maxIter = 8; end
if nargin < 4, base_lambda = 0.5; end

n = size(F, 1);
c = size(F, 2);
ideal_size = n / c;

% 初始统计量
fKf = sum(F .* (K * F))';   % 每个簇内的 K 总和
ff  = sum(F)';              % 每个簇的大小
m_all = vec2ind(F')';       % 每个样本当前的簇编号

for it = 1:maxIter
    converged = true;

    % 当前簇大小和不平衡度
    cluster_sizes = sum(F)';
    max_ratio = max(cluster_sizes) / n;

    % 自适应调整平衡系数：
    %   平衡数据: 惩罚系数大 -> 偏向均衡
    %   不平衡数据: 惩罚系数小 -> 允许不均衡
    if max_ratio < 1.2 * ideal_size / n % 接近平衡
        lambda = base_lambda * 1.5;
    elseif max_ratio > 2.0 * ideal_size / n % 极不平衡
        lambda = base_lambda * 0.3;
    else
        lambda = base_lambda;
    end

    % ========= 批量计算推荐迁移目标 =========
    moves = m_all; % 默认保持原簇
    for i = 1:n
        m = m_all(i);
        if ff(m) <= 1
            continue; % 不拆掉只有1人的簇
        end

        % 计算迁入和迁出的目标值变化
        Y_A = F' * K(:, i);

        fKf_s = fKf + 2 * Y_A + K(i, i);
        fKf_s(m) = fKf(m);
        ff_k = ff + 1;
        ff_k(m) = ff(m);

        fKf_0 = fKf;
        fKf_0(m) = fKf(m) - 2 * Y_A(m) + K(i, i);
        ff_0 = ff;
        ff_0(m) = ff(m) - 1;

        delta = fKf_s ./ ff_k - fKf_0 ./ ff_0;

        % ====== 自适应软约束惩罚（只影响选择，不影响目标值）======
        penalty = lambda * log((ff_k) / ideal_size + eps);
        delta_adj = delta + penalty;

        % 找最小的 delta_adj
        [delta_min, p] = min(delta_adj);

        if p ~= m && delta_min < -1e-8
            moves(i) = p; % 暂存迁移目标
        end
    end

    % ========= 批量执行迁移 =========
    for i = 1:n
        m = m_all(i);
        p = moves(i);
        if p ~= m
            converged = false;
            % 更新簇统计
            Y_A = F' * K(:, i);

            fKf(m) = fKf(m) - 2 * Y_A(m) + K(i, i);
            fKf(p) = fKf(p) + 2 * Y_A(p) + K(i, i);
            ff(m) = ff(m) - 1;
            ff(p) = ff(p) + 1;

            % 更新F矩阵
            F(i, m) = 0;
            F(i, p) = 1;
            m_all(i) = p;
        end
    end

    if converged
        break;
    end
end

% 返回最终目标值
obj = sum(fKf ./ ff);
end
