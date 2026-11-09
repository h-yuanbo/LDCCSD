function nbrIdx = build_knn_graph_rbf_kernel(X, k, sigma)
    % build_knn_graph_rbf_kernel: 使用RBF核函数构建k-NN图
    %
    % 输入:
    %   X     : n x d 的数据矩阵，n 是样本数，d 是特征维度。
    %   k     : 每个样本要找的最近邻居的数量。
    %   sigma : RBF核函数的带宽参数 (sigma > 0)。它控制了相似度衰减的速度。
    %           sigma 越大，相似度衰减越慢，更多远的点被认为是相似的。
    %           sigma 越小，相似度衰减越快，只有非常近的点才被认为是相似的。
    %           这是RBF核最关键的参数，通常需要通过交叉验证或经验值来选择。
    %
    % 输出:
    %   nbrIdx: n x k 的整数索引矩阵，nbrIdx(i,:) 存储了样本 i 的 k 个最相似邻居的索引。

    if sigma <= 0
        error('sigma 参数必须大于 0。');
    end

    n = size(X, 1); % 样本数量

    % 1. 计算所有样本对之间的平方欧氏距离
    % D_sq(i,j) 存储了样本 i 和样本 j 之间欧氏距离的平方。
    D_sq = pdist2(X, X, 'squaredeuclidean');

    % 2. 应用 RBF 核函数计算相似度矩阵 S
    % S(i,j) = exp(- ||xi - xj||^2 / (2 * sigma^2))
    % 相似度值范围在 [0, 1] 之间，1 表示完全相同，0 表示非常不相似。
    S = exp(-D_sq / (2 * sigma^2));

    % 3. 排除自身相似度 (将对角线元素设为0或一个非常小的值)
    % 这样在排序时就不会把样本自身选为邻居。
    % 注意：如果算法允许自环，可以跳过此步，但KNN图通常不包含自环。
    S(1:n+1:end) = 0; % 将对角线元素设为 0

    % 4. 对相似度矩阵 S 的每一行进行降序排序
    % 因为我们找的是“最相似”的，所以是降序。
    % `order` 矩阵存储了排序后的索引。
    [~, order] = sort(S, 2, 'descend');

    % 5. 提取每个样本的 k 个最相似邻居的索引
    % nbrIdx(i, :) 包含了样本 i 的 k 个最近邻居的索引。
    nbrIdx = order(:, 1:k);

end
