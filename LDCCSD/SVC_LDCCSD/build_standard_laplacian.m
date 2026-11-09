function L = build_standard_laplacian(X, H, nbrIdx, gamma)
    % 构建标准拉普拉斯矩阵（所有权重为1，对应目标函数1.x系列）
    [n, ~] = size(X);
    L = zeros(n, n);
    
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Ki = Xi * Xi';  % 标准K矩阵
        Mi = Ki + gamma * eye(size(Ki,1));
        HMH = H * (Mi \ H);
        L(Ni, Ni) = L(Ni, Ni) + HMH;  % 权重为1
    end
end