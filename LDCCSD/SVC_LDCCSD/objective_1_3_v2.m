function [labels_pred, Y, history] = objective_1_3_v2(X, Y_true, params)
% ------------------------------------------------------------
% Algorithm 1.2: 基于3.2模板重写
% 目标函数1.2: min_G sum_i tr[G^{iT} H^i (X^{iT}X^i + γI)^{-1} H^i Y^i (Y^{iT}Y^i)^{-1/2}]
% ------------------------------------------------------------

% ---- 基本量 ----
n = size(X, 1);
c = params.nCluster;

L = build_LDMGI_style_laplacian(X, params.k, params.gamma);
[V, eigenvals] = eig(full(L));
[~, idx] = sort(diag(eigenvals), 'ascend');
    
% 选择最小的c个特征值对应的特征向量（跳过第一个，通常是0）
G = real(V(:, idx(2:c+1)));

[labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 10);
% idx2 = randi(n, ceil(0.3*n));
% labels_pred(idx2) = randi(c, ceil(0.3*n));

Y = ind2vec(labels_pred')';
% ---- 初始化 ----
% lab = randi(c, [n,1]);
% Y = full(sparse((1:n)', lab, 1, n, c));
result = my_eval_y(labels_pred, Y_true);
disp(['Algorithm 1.3 init: Acc=', num2str(result.Accuracy)]);


% ---- 记录器 ----
history.obj = zeros(params.maxIter, 1);
history.acc = zeros(params.maxIter, 1);

% ---- 主循环 ----
for it = 1:params.maxIter

   
    [Y, obj_F] = solve_Y(L, Y);

    [~, labels_pred] = max(Y, [], 2);
    result = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 1.3  iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(obj_F(end))]);

    % (3) 更新G：使用SVD方法保证正交性
    
    % 计算目标函数值
    % fval = trace(G' * S_Y);

    % 记录
     history.obj(it) = obj_F(end);
     history.acc(it) = result.Accuracy;
end
counts = histcounts(labels_pred, 1:c+1);
fprintf('最终每个簇的样本数:\n');
disp(counts);
figure; plot(history.obj, 'LineWidth',1.5);
    xlabel('Iteration'); ylabel('Objective'); title(['Objective trajectory ', version]);
% 输出类别分布
% fprintf('Y的类别分布：\n');
% for j = 1:c
%     count = full(sum(Y(:, j)));
%     fprintf('类别%d: %d个样本 (%.1f%%)\n', j, count, 100*count/n);
% end
% 
% disp(result);
end