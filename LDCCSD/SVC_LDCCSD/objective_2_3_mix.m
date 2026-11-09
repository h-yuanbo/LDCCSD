function [labels_pred,history] = objective_2_3_mix(X, Y_true, params)
    % Algorithm 2.3: Self-paced quadratic form in Y only
    % 对应目标函数2.3: min_{u,Y} sum_i u_i * tr[(Y^{iT}Y^i)^{-1/2} Y^{iT} H^i (...) H^i Y^i (Y^{iT}Y^i)^{-1/2}] - τ sum_i u_i
    
    
    n = size(X, 1);                    % 样本数
    c = params.nCluster;               % 聚类类别数
    k = params.k;                       % k近邻参数
    H = eye(k) - (1/k)*ones(k);        % 中心化矩阵
    tau = params.tau;% 初始 SPL 门槛 
    tau_prime=params.tau_prime;
    taudelt = params.taudelt;       % 每轮递增系数
    taudelt_prime = params.taudelt_prime; 
    tau_max=params.tau_max;
    nbrIdx = build_knn_graph(X, k);
    % ==== 初始化 ====
    u = zeros(n, 1); 
    L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    % 谱初始化聚类
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',1);
    idx2 = randi(n, ceil(0.3*n));
    labels_pred(idx2) = randi(c, ceil(0.3*n));
    Y = full(ind2vec(labels_pred')');
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3f1 init: Acc=', num2str(result_init.Accuracy)]);
    tol = 1e-12; 
%     trace_vec_init = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    % 历史记录
    history.obj = zeros(params.maxIter, 1);
    history.beta = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.Accuracy=zeros(params.maxIter, 1);

 
    for it = 1:params.maxIter
        % (1) 更新Y - 主要优化变量
        L_u= assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma); 
        %fprintf('before update Y compute_trace');
        %trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        [Y_hat, obj_F] = solve_Y(L_u, Y);
        alpha = 0.9;  % 例如
        Y = (1 - alpha)*Y + alpha*Y_hat;
  
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
%         figure; 
%         histogram(trace_vec, 50);
%         title(' Loss after Y Distribution');
%         xlabel('Loss value'); 
%         ylabel('Count');
        u = mixture_weighting_trace(trace_vec, tau, tau_prime,u);
%         fval = sum(u .* trace_vec) +((0.5 * (u.'*u) - sum(u))*tau);
        %fval = sum(u .* trace_vec) - tau * sum(u);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 2.3  iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(obj_F(end))]);
        % 记录
        history.obj(it) = obj_F(end);
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
        history.Accuracy(it)=result.Accuracy;
        if tau<tau_max
            tau = tau * taudelt;
            tau_prime=tau_prime*taudelt_prime;
        end
    end
    counts = histcounts(labels_pred, 1:c+1);
    fprintf('最终每个簇的样本数:\n');
    disp(counts);
    plot_history(history, '2.3');
end
