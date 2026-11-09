function G = update_G_12(X, nbrIdx, H, Y, gamma, epsY)
    % 按照1.2目标函数更新G
    [n, c] = size(Y);
    k = size(H, 1);
    S = zeros(n, c);
    
    for i = 1:n
        idx = nbrIdx(i, :);               % 第i个样本的k个邻居索引
        Xi = X(idx, :);                   % k×d 邻域特征矩阵
        Ki = Xi * Xi';                    % k×k == X^{iT} X^i
        
        % 计算 (X^{iT}X^i + γI)^{-1}
        Ai = inv(Ki + gamma * eye(k));
        
        % 提取邻域对应的Y
        Yi = Y(idx, :);                   % k×c
        
        % 计算 (Y^{iT}Y^i)^{-1/2} 并正则化
        YtY = Yi' * Yi;                   % c×c
        YtY_reg = YtY + epsY * eye(c);
        YtY_inv_sqrt = YtY_reg^(-0.5);
        
        % 检查数值稳定性
        if any(isnan(YtY_inv_sqrt(:))) || any(isinf(YtY_inv_sqrt(:)))
            warning('数值不稳定在样本 %d, 使用单位矩阵', i);
            YtY_inv_sqrt = eye(c);
        end
        
        % 累加到S矩阵：H^i * A^i * H^i * Y^i * (Y^{iT}Y^i)^{-1/2}
        contribution = H * Ai * H * Yi * YtY_inv_sqrt;
        S(idx, :) = S(idx, :) + contribution;
    end
    
    % 求解 min_{G^T G = I} tr(G^T S) 使用SVD
    [U, ~, V] = svd(S, 'econ');
    G = -U * V';  % 负号确保最小化
    
    % 确保正交性
    G = orth(G);
end