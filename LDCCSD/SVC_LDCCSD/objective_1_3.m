function [labels_pred, history] = objective_1_3(X, Y_true, params)
    % Algorithm 1.3: Quadratic form in Y only
    % 对应目标函数1.3: min_Y sum_i tr[(Y^{iT}Y^i)^{-1/2} Y^{iT} H^i (...) H^i Y^i (Y^{iT}Y^i)^{-1/2}]
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    
    nbrIdx = build_knn_graph(X, k);
    
    % 初始化
    lab = randi(c, [n,1]);
    Y = full(sparse((1:n)', lab, 1, n, c));
    
    result_init = my_eval_y(lab, Y_true);
    disp(['Algorithm 1.3 init: Acc=', num2str(result_init.Accuracy)]);
    
    history.obj = zeros(params.maxIter, 1);
    
    for it = 1:params.maxIter
        % 构建标准拉普拉斯矩阵
        L = build_standard_laplacian(X, H, nbrIdx, params.gamma);
        
        % 从Y构造G（目标函数1.3中G不是独立变量）
        G = Y * diag(sum(Y,1).^(-0.5));
        
        % 更新Y
        Q = L * G;
        [Y, obj_F] = update_Y_min(Q, Y, 1);
        
        % 计算目标函数值
        fval = compute_objective_1_3(X, Y, nbrIdx, H, params.gamma);
        
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 1.3 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        
        history.obj(it) = fval;
    end
    
    disp(result);
    plot_historywithoutu(history, '1.3');
end
