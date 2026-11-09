function L = assemble_L_for_1_3(X, H, nbrIdx, gamma)
% 构造 1.3 的拉普拉斯矩阵 L
% X: n×d 数据矩阵（每行是样本）
% H: k×k 中心化矩阵（H = eye(k) - ones(k)/k）
% nbrIdx: n×k 邻居索引矩阵（每行是样本 i 的 k 个邻居）
% gamma: 正则化参数
%
% 输出 L: n×n 拉普拉斯矩阵（对称）

    n = size(X,1);  
    k = size(H,1);  
    L = zeros(n,n); % 初始化

    for i = 1:n
        Ni = nbrIdx(i,:);             % 样本 i 的邻居索引
        Xi = X(Ni,:);                  % 邻居的特征
        Ki = Xi * Xi';                 % 局部 Gram 矩阵
        Mi = Ki + gamma * eye(k);      % 加 γI 做正则
        % 局部平滑矩阵（对应公式里的 H^i (X^{iT}X^i+γI)^{-1} H^i）
        Llocal = H * (Mi \ H);         
        % 累加到全局 L
        L(Ni,Ni) = L(Ni,Ni) + Llocal;
    end

    % 保证对称性
    L = 0.5 * (L + L');  
end
