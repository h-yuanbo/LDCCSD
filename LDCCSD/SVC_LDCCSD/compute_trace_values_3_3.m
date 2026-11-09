function trace_values = compute_trace_values_3_3(X, Y, u, nbrIdx, H, gamma)
% 计算 3.3 公式中每个样本的个体损失 trace_i
%
% X       : n × d 数据矩阵
% Y       : n × c one-hot 标签
% u       : n × 1 样本权重 [0,1]
% nbrIdx  : n × k 邻居索引矩阵（每行是该样本的 k 个邻居ID）
% H       : k × k 中心化矩阵
% gamma   : 正则化参数（≥0）
%
% trace_values : n × 1 向量，每个元素是该样本的 trace 损失

    n = size(X, 1);
    k = size(nbrIdx, 2);
    trace_values = zeros(n, 1);
    
    for i = 1:n
        try
            % ---- 获取邻域信息 ----
            nbr_ids = nbrIdx(i, :);          % k × 1
            X_i = full(X(nbr_ids, :));       % k × d
            Y_i = full(Y(nbr_ids, :));       % k × c
            u_i = full(u(nbr_ids));          % k × 1
            
            % ---- Step 1: 计算稳定的 YtY 和 sqrt inverse ----
            YtY = full(Y_i' * Y_i);
            % SVD 安全逆开方
            [UY, SY, VY] = svd(YtY);
            sY = diag(SY);
            sY(sY < 1e-10) = 1e-10;
            YtY_neg_half = VY * diag(1 ./ sqrt(sY)) * UY';
            
            % ---- Step 2: 构造加权邻域 Gram 矩阵 ----
            % 权重对特征进行缩放
            WXi = diag(u_i) * X_i;    % k × d
            G = full(WXi * WXi');     % k × k
            G_reg = G + gamma * eye(k);
            
            % ---- Step 3: 稳定求逆 ----
            % 用 SVD 或 pinv 避免奇异
            if rcond(G_reg) < 1e-12
                Ginv = pinv(G_reg);
            else
                Ginv = inv(G_reg);
            end
            
            % ---- Step 4: 核心 trace 计算 ----
            middle_term = Y_i' * H * Ginv * H * Y_i;
            loss_matrix = YtY_neg_half * middle_term * YtY_neg_half;
            
            trace_values(i) = real(trace(loss_matrix));
            
        catch ME
            warning('计算样本 %d 损失失败, 使用默认值: %s', i, ME.message);
            trace_values(i) = 0; % 或者设一个默认大值，比如 max(trace_values)
        end
    end
end
