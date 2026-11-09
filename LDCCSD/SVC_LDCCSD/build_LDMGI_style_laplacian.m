function L = build_LDMGI_style_laplacian(X, k, lamda)
    % 按照LDMGI算法构建拉普拉斯矩阵
    [n, d] = size(X);
    X = X';  % 转置为d×n，每列是一个数据点（LDMGI格式）
    
    % 中心化矩阵
    Lc = eye(k) - 1/k*ones(k);
    
    % 初始化稀疏矩阵
    A = spalloc(n*k, n*k, 5*n*k);
    S = spalloc(n, n*k, 5*n*k);
    
    for i = 1:n
        % 计算到所有点的距离
        dis = sum(bsxfun(@minus, X(:,i), X).^2);
        [~, nnidx] = sort(dis);
        
        % 获取k个最近邻
        Xi = X(:, nnidx(1:k));
        Xi = Xi * Lc;  % 中心化
        
        % 根据维度选择数值稳定的求逆方法
        if d > k
            % 高维情况：直接求逆
            Ai = inv(lamda*eye(k) + Xi'*Xi);
            Ai = Lc * Ai * Lc;
        else
            % 低维情况：使用Woodbury矩阵恒等式
            Ai = Lc - lamda*Xi' * inv(eye(d) + lamda*Xi*Xi') * Xi;
        end
        
        % 将局部矩阵放入全局稀疏矩阵
        lidx = (i-1)*k+1:i*k;
        A(lidx, lidx) = Ai;
        S(nnidx(1:k), lidx) = eye(k);
    end
    
    % 构建最终的拉普拉斯矩阵
    L = S * A * S';
    
    % 确保对称性
    L = (L + L') / 2;
end