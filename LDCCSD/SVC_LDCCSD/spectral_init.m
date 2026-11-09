function [Y0, Urow, Lspec] = spectral_init(X, c, k, sigma)
% X: n×d (建议先 z-score 标准化)
% c: 簇数, k: knn, sigma: 热核宽度(可取中位数距离/√d)

n = size(X,1);

% 1) kNN + 热核相似度
[idx, dist] = knnsearch(X, X, 'K', k+1);  % 自身在第1列
idx  = idx(:,2:end);
dist = dist(:,2:end);
W = sparse(repmat((1:n)',1,k), idx, exp(-(dist.^2)/(sigma^2)), n, n);
W = max(W, W');                            % 对称化

% 2) 归一化拉普拉斯
d = sum(W,2);
Dinv2 = spdiags(1./sqrt(d+eps), 0, n, n);
Lspec = speye(n) - Dinv2*W*Dinv2;         % L_sym

% 3) 最小的 c 个特征向量
opts.disp = 0; opts.issym = 1; opts.isreal = 1;
[U, ~] = eigs(Lspec, c, 'smallestreal', opts);   % n×c

% 4) 行归一化
Urow = bsxfun(@rdivide, U, sqrt(sum(U.^2,2))+eps);

% 5) k-means 得到初始标签 & Y0
lab0 = kmeans(Urow, c, 'Replicates', 10, 'MaxIter', 200);
Y0 = full(sparse((1:n)', lab0, 1, n, c));end
