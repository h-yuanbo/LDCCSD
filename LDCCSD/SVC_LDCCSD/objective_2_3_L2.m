function [labels_pred,history] = objective_2_3_L2(X, Y_true, params)
    % Algorithm 2.3 with L2 regularization on u
    % 原目标: min_{u,Y} sum_i u_i * trace(A_i) - tau sum_i u_i
    % 改为:   min_{u,Y} sum_i u_i * trace(A_i) + lambda * sum_i u_i^2

    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    sigma = params.sigma;
    lambda = 0.1; % L2 正则化强度，可调节

    % 构建k近邻图
    nbrIdx = build_knn_graph_rbf_kernel(X, k, sigma);

    % 初始化 u
    u = ones(n, 1);

    % 初始化 Laplacian & 特征分解
    L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);

    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');

    % 选择最小的 c 个非零特征值对应的特征向量
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 100);
    idx2 = randi(n, ceil(0.3*n));
    labels_pred(idx2) = randi(c, ceil(0.3*n));
    % 初始化 Y
    Y = ind2vec(labels_pred')';
    Y = full(Y);

    % 评估初始结果
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 L2 init: Acc = ', num2str(result_init.Accuracy)]);

    % 历史记录
    history.obj = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);

    fval = 0;

    % ============= 主循环 =============
    for it = 1:params.maxIter
        % (1) 更新 Y
        L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        [Y, obj_F] = solve_Y(L_u, Y);
        [~, labels_pred] = max(Y, [], 2);

        result = my_eval_y(labels_pred, Y_true);
        disp(['Iter ', num2str(it), ...
              ': Acc = ', num2str(result.Accuracy), ...
              '    Obj = ', num2str(fval)]);

        % (2) 计算 trace 值（对应 A_i）
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);

        % (3) L2 正则闭式更新公式: u = -A_i/(2λ)
        u = 1 ./ (1 + exp((trace_vec - params.tau) / lambda));
        % 投影到 [0,1]
        u = max(0, min(1, u));

        % (4) 计算目标函数值
        fval = sum(u .* trace_vec) + lambda * sum(u.^2);

        % (5) 记录
        history.obj(it) = fval;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
    end

    % 绘制收敛曲线
    plot_history(history, '2.3 L2');
end
