function [Y, lab, F] = spectral_init_from_L(L, c)
    % L: n×n，对称的“拉普拉斯型”矩阵（半正定，越小越近）
    % c: 簇数
    n = size(L,1);
    L = (L+L')/2;                         % 数值对称

    % 取最小的 c 个特征向量
    [F,~] = eigs(L, c, 'smallestreal');

    % 行归一化（Ncut 标配，防止某些样本行范数过大支配）
    F = F ./ (sqrt(sum(F.^2,2)) + 1e-12);

    % k-means 离散化
    opts = {'Replicates',20,'MaxIter',300,'Start','plus', ...
            'EmptyAction','singleton','Display','off'};
    lab = kmeans(F, c, opts{:});

    % 变一热
    Y = sparse((1:n)', lab, 1, n, c);
end
