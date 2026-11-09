function [labels_pred, history] = objective_2_1(X, Y_true, params)
    % Algorithm 2.1: Self-paced spectral clustering (without correlation weights)
    % 对应目标函数2.1: min_{u,G} sum_i u_i * tr[G^{iT} H^i (X^{iT}X^i + γI)^{-1} H^i G^i] - τ sum_i u_i
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    
    nbrIdx = build_knn_graph(X, k);
    
    % 初始化
    u = 0.1 + 0.15*rand(n, 1);
    lab = randi(c, [n,1]);
    Y_init = full(sparse((1:n)', lab, 1, n, c));
    G = Y_init * diag(sum(Y_init,1).^(-0.5));
    
    result_init = my_eval_y(lab, Y_true);
    disp(['Algorithm 2.1 init: Acc=', num2str(result_init.Accuracy)]);
    
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    
    for it = 1:params.maxIter
        % (1) 更新L_u - 标准拉普拉斯矩阵
        L_u = assemble_Lu_standard(X, H, nbrIdx, u, params.gamma);
        
        % (2) 更新G - 特征分解
        [G, ~] = eigs(L_u, c, 'smallestreal');
        
        % (3) 更新u - 解析解（阈值规则）
        trace_vec = compute_trace_values(X, G, nbrIdx, H, params.gamma);
       % 软阈值自步学习
        u = max(0, 1 - trace_vec / tau);  % 线性递减
        u = max(u, 0.01);  % 最小值保护

       
        
        fval = sum(u .* trace_vec) - tau * sum(u);
        
        % (4) 得到聚类结果
        G2 = bsxfun(@rdivide, G, sqrt(sum(G.^2, 2)) + eps);
        G2 = real(G2); 
        labels_pred = kmeans(G2, c, 'Replicates', 10, 'MaxIter', 100);
        
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 2.1 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        
        % 记录
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) median(u) max(u)];
        
        tau = tau * 1.1;
    end
    
    disp(result);
   % plot_history(history, '2.1');
end
