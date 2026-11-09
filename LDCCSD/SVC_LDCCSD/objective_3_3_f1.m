function [labels_pred, history] = objective_3_3_f1(X, Y_true, params)
    % Algorithm 3.3: Correlated self-paced quadratic form in Y only
    % 对应目标函数3.3: min_{u,Y} sum_i u_i * tr[(Y^{iT}Y^i)^{-1/2} Y^{iT} H^i (...) H^i Y^i (Y^{iT}Y^i)^{-1/2}] - τ sum_i u_i
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    taudelt=params.taudelt;
   nbrIdx = build_knn_graph(X, k);
    % 初始化
    %u = 0.15*rand(n, 1);
    u =ones(n, 1);
    L_u =  assemble_Lu(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    
    % 选择最小的c个特征值对应的特征向量（跳过第一个，通常是0）
    G = real(V(:, idx(2:c+1)));
    % K = exp(-5 * L);
    % [G] = fast_spectral_clustering(K, c);

    
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 100);
    idx2 = randi(n, ceil(0.3*n));
    labels_pred(idx2) = randi(c, ceil(0.3*n));

    Y = ind2vec(labels_pred')';
    
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 3.3 init: Acc=', num2str(result_init.Accuracy)]);
    
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);

    
    for it = 1:params.maxIter

        
        % (1) 更新Y
        L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);

        [Y, obj_F] = solve_Y(L_u, Y);
        Y = full(Y);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 3.3 iter-a', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(obj_F(end))]);
        
        % (2) 更新u
        %G = Y * diag(sum(Y,1).^(-0.5));
      options = optimoptions('fmincon', ...
            'SpecifyObjectiveGradient', true, ...
            'Algorithm','sqp', ...
            'HessianApproximation','lbfgs', ...
            'MaxIterations', 20, ...
            'Display','none', ...
            'StepTolerance', 1e-6, ...
            'OptimalityTolerance', 1e-6);

        [u, tau, fval] = update_u_fixed_f1(u, X', Y',nbrIdx', H, params.gamma, tau, options);
        
        
        % 记录
        history.obj(it) = obj_F(end);
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];


            tau = tau * taudelt;
    

    end
    
    [~, labels_pred] = max(Y, [], 2);
    %plot_history(history, '3.3f1');
 end
