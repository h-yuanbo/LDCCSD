function Y = update_Y_12(X, nbrIdx, H, G, Y_prev, gamma, epsY)
    % 修正的Y更新版本
    [n, c] = size(Y_prev);
    k = size(H, 1);
    
    % 构造成本矩阵
    C = zeros(n, c);
    
    for i = 1:n
        idx = nbrIdx(i, :);
        Xi = X(idx, :);
        Ki = Xi * Xi';
        
        % 检查数值稳定性
        if rcond(Ki + gamma * eye(k)) < 1e-12
            Ai = pinv(Ki + gamma * eye(k));
        else
            Ai = inv(Ki + gamma * eye(k));
        end
        
        % 计算 H * Ai * H (k×k矩阵)
        HAH = H * Ai * H;
        
        % 对每个可能的类别分配计算成本
        for j = 1:c
            % 创建一个测试的Y_i，假设当前样本分配到类别j
            Y_test_i = zeros(k, c);  % k个邻居×c个类别
            
            % 将邻域的实际标签复制过来
            Y_test_i = Y_prev(idx, :);  % k×c
            
            % 计算(Y^T*Y)^{-1/2}
            YtY = Y_test_i' * Y_test_i + epsY * eye(c);  % c×c
            if rcond(YtY) < 1e-12
                YtY_inv_sqrt = eye(c);
            else
                YtY_inv_sqrt = YtY^(-0.5);
            end
            
            % 计算这个类别分配的成本
            % tr[G^{iT} * HAH * Y_i * (Y^T*Y)^{-1/2}]
            G_i = G(idx, :);  % k×c
            term = G_i' * HAH * Y_test_i * YtY_inv_sqrt;  % c×c
            C(i, j) = trace(term);
        end
    end
    
    % 选择最小成本的类别（因为是最小化问题）
    [~, labels] = min(C, [], 2);
    
    % 确保labels有效
    labels = max(1, min(labels, c));
    
    % 构造Y矩阵
    Y = zeros(n, c);
    for i = 1:n
        Y(i, labels(i)) = 1;
    end
end
