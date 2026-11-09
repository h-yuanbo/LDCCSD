function [labels_pred, history] = objective_3_3_f2(X, Y_true, params)
    n = size(X, 1);                    % 样本数
    c = params.nCluster;               % 聚类类别数
    k = params.k;                       % k近邻参数
    H = eye(k) - (1/k)*ones(k);        % 中心化矩阵
    tau = params.tau;            % 初始 SPL 门槛 
    taudelt = params.taudelt;       % 每轮递增系数
    tau_max=params.tau_max;
    nbrIdx = build_knn_graph(X, k);
    % ==== 初始化 ====
    u = 0.15*rand(n, 1);
    L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
    % 谱初始化聚类
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',1);
    idx2 = randi(n, ceil(0.3*n));
    labels_pred(idx2) = randi(c, ceil(0.3*n));
    Y = full(ind2vec(labels_pred')');
    result_init = my_eval_y(labels_pred, Y_true);
    tol = 1e-12; 
%     trace_vec_init = compute_trace_values_3_3(X, Y, nbrIdx, H, params.gamma);
%     idx = trace_vec_init < tau;                 % 满足条件的样本
%     u(idx) = 1 - (1/tau) * trace_vec_init(idx); % 公式里的 1 - (1/tau)*trace
    history.obj = zeros(params.maxIter, 1);
    history.beta = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.Accuracy=zeros(params.maxIter, 1);

    
    % ==== 正式迭代 ====
    for it = 1:params.maxIter
        % (1) 更新Y
        L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
        [Y, obj_F] = solve_Y(L_u, Y);
        Y = full(Y);
 %       trace_vec = compute_trace_values_3_3(X, Y, u, nbrIdx, H, params.gamma);        
        % (2) 更新u
        options = optimoptions('fmincon', ...
            'SpecifyObjectiveGradient', true, ...
            'Algorithm','sqp', ...
            'HessianApproximation','lbfgs', ...
            'MaxIterations', 20, ...
            'Display','none', ...
            'StepTolerance', 1e-6, ...
            'OptimalityTolerance', 1e-6);
        
        [u, tau, fval] = update_u_fixed(u, X', Y', nbrIdx', H, params.gamma, tau, options);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 3.3 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        % 记录
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
        
        % (3) 外层tau更新
        tau = tau * taudelt;
    end
    
    [~, labels_pred] = max(Y, [], 2);
    plot_history(history, '3.3');
end
