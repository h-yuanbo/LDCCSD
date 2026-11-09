function [labels_pred,history] = objective_3_1(X, Y_true, params)
    % Algorithm 3.1: Correlated self-paced (只优化G和u，没有Y)
    % 对应目标函数3.1: min_{u,G} sum_i u_i * tr[G^{iT} H^i (diag(u^i) X^{iT}X^i diag(u^i) + γI)^{-1} H^i G^i] - τ sum_i u_i
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    nbrIdx = build_knn_graph(X, k);
    
    % 初始化（与LDCCSD一致）
    u = rand(n,1);
    idx = u > 0.7;
    u(idx) = 0.01;
    
    % 不需要初始化Y，因为目标函数3.1中没有Y
    % 初始化G
    lab = randi(c, [n,1]);
    Y_init = full(sparse((1:n)', lab, 1, n, c));
    
    
    % 用你的assemble_Lu初始化
    %L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
    result_init = my_eval_y(lab, Y_true);
    G = Y_init * diag(sum(Y_init,1).^(-0.5)); % 只用于G的初始化
    disp(['Algorithm 3.1 init: Acc=', num2str(result_init.Accuracy)]);
    
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    fval=0;
    for it = 1:params.maxIter
        % 更新L_u（使用相关性）
        L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
        
        % 更新G（目标函数3.1的主要优化变量）
        [G, ~] = eigs(L_u, c, 'smallestreal');
        
        % 从G得到聚类结果（Y不是优化变量，只是输出表示）
        G2 = bsxfun(@rdivide, G, sqrt(sum(G.^2, 2)) + eps);
        labels_pred = kmeans(G2, c, 'Replicates', 10, 'MaxIter', 100);
        
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 3.1 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        % 更新u（使用你的update_u_fixed函数）
        options = optimoptions('fmincon', ...
            'SpecifyObjectiveGradient', true, ...
            'Algorithm','sqp', ...
            'HessianApproximation','lbfgs', ...
            'Display','none', ...
            'MaxIterations', params.innerU_MaxIters, ...
            'StepTolerance',1, ...
            'OptimalityTolerance',1e-2);
        
        [u, tau, fval] = update_u_fixed(u, X', G', nbrIdx', H, params.gamma, ...
                                       tau, params.tau_beta, options, it, params.maxIter);
        
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) median(u) max(u)];
        
        
        
        tau = tau * 1.01;
    end
    
    % 最终输出
    G2 = bsxfun(@rdivide, G, sqrt(sum(G.^2, 2)) + eps);
    labels_pred = kmeans(G2, c, 'Replicates', 10, 'MaxIter', 100);

    disp(result);

    figure; plot(history.obj, 'LineWidth',1.5);
    xlabel('Iteration'); ylabel('Objective'); title('Objective trajectory3.1');

    figure; plot(history.u_stat, 'LineWidth',1.5);
    xlabel('Iteration'); ylabel('u values');
    legend('min(u)','median(u)','max(u)');
    title('Evolution of u distribution3.1');
end
