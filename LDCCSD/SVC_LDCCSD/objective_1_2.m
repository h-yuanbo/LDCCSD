function [labels_pred, Y, history] = objective_1_2(X, Y_true, params)
% ------------------------------------------------------------
% Algorithm 1.2: 基于3.2模板重写
% 目标函数1.2: min_G sum_i tr[G^{iT} H^i (X^{iT}X^i + γI)^{-1} H^i Y^i (Y^{iT}Y^i)^{-1/2}]
% ------------------------------------------------------------

% ---- 基本量 ----
n = size(X, 1);
c = params.nCluster;
k = params.k;
H = eye(k) - (1/k)*ones(k);
nbrIdx = build_knn_graph(X, k);


options = [];
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';
% options.t = 1;
S = constructW(X, options);
[G, accHistory] = fast_spectral_clustering(S, c, Y_true);

[labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 10);
Y = ind2vec(labels_pred')';
% ---- 初始化 ----
% lab = randi(c, [n,1]);
% Y = full(sparse((1:n)', lab, 1, n, c));
result = my_eval_y(labels_pred, Y_true);
disp(['Algorithm 1.2 v5 init: Acc=', num2str(result.Accuracy)]);

% G = Y * diag(sum(Y,1).^(-0.5));
% fval = 0;

% ---- 记录器 ----
history.obj = zeros(params.maxIter, 1);
history.acc = zeros(params.maxIter, 1);

L = build_LDMGI_style_laplacian(X, k, params.gamma);

% ---- 主循环 ----
for it = 1:params.maxIter
    
    % (1) 构建目标函数1.2对应的矩阵S
    S_G = G' * L;
    % S_G = build_objective_12_matrix_left(X, G, nbrIdx, H, params.gamma);
    
    % (2) 更新Y：使用与3.2相同的稳定更新方式
    innerIterY = 10;
    [Y, obj_F] = update_Y_min(S_G', Y, innerIterY);
    
    [~, labels_pred] = max(Y, [], 2);
    result = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 1.2 v5 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(obj_F(end))]);
    
    % (3) 更新G：使用SVD方法保证正交性
    % S_Y = build_objective_12_matrix_right(X, Y, nbrIdx, H, params.gamma);
    S_Y = L * Y * diag(1./sqrt(sum(Y,1)+eps));
    [U, ~, V] = svd(-S_Y, 'econ');
    G = U * V';
    if size(G, 2) > c
        G = G(:, 1:c);
    end
    G = orth(G);
    
    % 计算目标函数值
    fval = trace(G' * S_Y);
    
    % 记录
    history.obj(it) = fval;
    history.acc(it) = result.Accuracy;
end

% 输出类别分布
fprintf('Y的类别分布：\n');
for j = 1:c
    count = sum(Y(:,j));
    fprintf('类别%d: %d个样本 (%.1f%%)\n', j, count, 100*count/n);
end

disp(result);
end

function S = build_objective_12_matrix_right(X, Y, nbrIdx, H, gamma)
% 构建目标函数1.2对应的系数矩阵S
% 目标：sum_i tr[G^{iT} H^i (X^{iT}X^i + γI)^{-1} H^i Y^i (Y^{iT}Y^i)^{-1/2}]

n = size(X, 1);
c = size(Y, 2);
k = size(nbrIdx, 2);
S = zeros(n, c);

for i = 1:n
    idx = nbrIdx(i, :);
    Xi = X(idx, :);  % k×d
    Yi = Y(idx, :);  % k×c
    
    % 计算 (X^{iT}X^i + γI)^{-1}
    % 注意：这里应该是Xi的Gram矩阵 k×k
    Ki = Xi * Xi' + gamma * eye(k);  % k×k
    if rcond(Ki) < 1e-12
        Ai = pinv(Ki);
    else
        Ai = inv(Ki);
    end
    %     s_diag = diag(S_y);
    %     s_diag(s_diag < 1e-12) = 1e-12;
    %     YiTYi_neg_half = U_y * diag(s_diag.^(-0.5)) * V_y';
    % end
    
    YiTYi_neg_half = diag(1./sqrt(sum(Y,1) + eps));

    % 计算 H^i * Ai * H^i * Y^i * (Y^{iT}Y^i)^{-1/2}
    % H: k×k, Ai: k×k, Yi: k×c, YiTYi_neg_half: c×c
    term = H * Ai * H * Yi * YiTYi_neg_half;  % k×c
    S(idx, :) = S(idx, :) + term;
end
end



function S = build_objective_12_matrix_left(X, G, nbrIdx, H, gamma)
% 构建目标函数1.2对应的系数矩阵S
% 目标：sum_i tr[G^{iT} H^i (X^{iT}X^i + γI)^{-1} H^i Y^i (Y^{iT}Y^i)^{-1/2}]

n = size(X, 1);
c = size(G, 2);
k = size(nbrIdx, 2);
S = zeros(n, c);

for i = 1:n
    idx = nbrIdx(i, :);
    Xi = X(idx, :);  % k×d
    Gi = G(idx, :);  % k×c
    
    % 计算 (X^{iT}X^i + γI)^{-1}
    % 注意：这里应该是Xi的Gram矩阵 k×k
    Ki = Xi * Xi' + gamma * eye(k);  % k×k
    if rcond(Ki) < 1e-12
        Ai = pinv(Ki);
    else
        Ai = inv(Ki);
    end
    

    term = Gi' * H * Ai * H;  % c×k
    S(idx, :) = S(idx, :) + term';
end
end