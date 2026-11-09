function [labels_pred,history] = objective_2_3_v0(X, Y_true, params)
    % Algorithm 2.3: Self-paced quadratic form in Y only
    % 对应目标函数2.3: min_{u,Y} sum_i u_i * tr[(Y^{iT}Y^i)^{-1/2} Y^{iT} H^i (...) H^i Y^i (Y^{iT}Y^i)^{-1/2}] - τ sum_i u_i
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    
    nbrIdx = build_knn_graph(X, k);
    
    % 初始化
    u =ones(n, 1);
    % lab = randi(c, [n,1]);
    % Y = full(sparse((1:n)', lab, 1, n, c));
    
    L = build_LDMGI_style_laplacian(X, k, params.gamma);
    [V, eigenvals] = eig(full(L));
    [~, idx] = sort(diag(eigenvals), 'ascend');

    % 选择最小的c个特征值对应的特征向量（跳过第一个，通常是0）
    G = real(V(:, idx(2:c+1)));
    % K = exp(-5 * L);
    % [G] = fast_spectral_clustering(K, c);

    
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
    idx2 = randi(n, ceil(0.3*n));
    labels_pred(idx2) = randi(c, ceil(0.3*n));

    Y = ind2vec(labels_pred')';

    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init: Acc=', num2str(result_init.Accuracy)]);
    
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    fval = 0;
    
    for it = 1:params.maxIter
        % (1) 更新Y - 主要优化变量
        L_u = assemble_Lu_standard(X, H, nbrIdx, u, params.gamma);
        [Y, obj_F] = solve_Y(L_u, Y);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 2.3 v5 iter-a', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        % G = Y * diag(sum(Y,1).^(-0.5));  % 从Y构造G
        % Q = L_u * G;
        % [Y, obj_F] = update_Y_min(Q, Y, 1);
        
        % (2) 更新u - 考虑双重归一化项
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        u = double(trace_vec < tau);
        u = max(u, 0.01);
        
        fval = sum(u .* trace_vec) - tau * sum(u);
        % 
        % [~, labels_pred] = max(Y, [], 2);
        % result = my_eval_y(labels_pred, Y_true);
        % disp(['Algorithm 2.3 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        
        % 记录
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) median(u) max(u)];
        %params.taudelt = 1.15;
        tau = tau * params.taudelt;
    end
    
    disp(result);
    plot_history(history, '2.3');
end
