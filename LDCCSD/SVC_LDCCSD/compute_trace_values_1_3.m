function trace_vec = compute_trace_values_1_3(X, Y, nbrIdx, H, gamma)
    % X: n×d 特征矩阵
    % Y: n×c one-hot标签矩阵
    % nbrIdx: n×k 邻居索引
    % H: 中心化矩阵 k×k
    % gamma: 正则化参数
    
    [n, ~] = size(X);
    k = size(H, 1);
    trace_vec = zeros(n, 1);
    
    for i = 1:n
        % 提取i的k个邻居索引
        neigh = nbrIdx(i, :); 
        
        % 邻域特征与标签
        Xi = X(neigh, :);        % k×d
        Yi = Y(neigh, :);        % k×c
        
        % 邻域内标签的协方差矩阵
        GY = Yi' * Yi;           % c×c
        GY_invhalf = inv(sqrtm(GY + 1e-8 * eye(size(GY)))); 
        
        % 邻域特征协方差（加γI正则）
        GX = Xi * Xi' + gamma * eye(k);
        GX_inv = inv(GX);
        
        % 按1.3公式计算trace损失
        temp = GY_invhalf * (Yi' * H * GX_inv * H * Yi) * GY_invhalf;
        trace_vec(i) = trace(temp);
    end
end