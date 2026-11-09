function [DU, accHistory] = fast_spectral_clustering(S, c, gt)
% Fast and Simple Spectral Clustering in Theory and Practice
% 
n = size(S, 1);
S = max(S, S');
d = sum(S,2);
D_inv_sqrt = spdiags(1 ./ sqrt(d + eps), 0, n, n);
% Normalized kernel
K = D_inv_sqrt * S * D_inv_sqrt;
K = (K + speye(n)) / 2;

U = randn(n, c);  % 随机初始化一个 n x c 的矩阵，每列是从 N(0, I) 中采样的高斯分布向量
t = 10 * ceil(log(n/c)); % 幂法的迭代次数

accHistory = [];
debug = false;
if exist('gt', 'var') && length(gt) == n
    debug = true;
end

if debug
    DU = D_inv_sqrt * U;
    [label, ~] = kmeanspp(DU', c);
    accHistory = compute_acc(gt, label);
end

for i = 1:t
    U = K * U;  % 对每一列进行幂法更新
    if debug
        DU = D_inv_sqrt * U;
        [label, ~] = kmeanspp(DU', c);
        accHistory = [accHistory; compute_acc(gt, label)]; %#ok
    end
end
DU = D_inv_sqrt * U;
end