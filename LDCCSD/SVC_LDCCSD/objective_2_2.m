function [labels_pred, Y, history] = objective_2_2_v2(X, Y_true, params)
    % Algorithm 2.2: 基于3.2模板，采用数值友好近似
    % 目标函数2.2: min_{u,G} sum_i u_i*tr[G^{iT} H^i (...)^{-1} H^i Y^i (Y^{iT}Y^i)^{-1/2}] - τ*sum_i u_i
    
    % 基本量
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    
    nbrIdx = build_knn_graph(X, k);
    
    % 初始化
    u = 0.1 + 0.15*rand(n, 1);
    lab = randi(c, [n,1]);
    Y = full(sparse((1:n)', lab, 1, n, c));
    result = my_eval_y(lab, Y_true);
    disp(['Algorithm 2.2 v2 init: Acc=', num2str(result.Accuracy)]);
    
    G = Y * diag(sum(Y,1).^(-0.5));
    fval = 0;
    
    % 记录器
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    
    % 主循环
    for it = 1:params.maxIter
        innerIterY = 1;
        
        % (1) 构建加权拉普拉斯L_u（借鉴3.2的assemble_Lu方法）
        L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
        
        % (2) 更新Y：使用稳定的update_Y_min
        Q = L_u * G;
        [Y, obj_F] = update_Y_min(Q, Y, innerIterY);
        
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 2.2 v2 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(fval)]);
        
        % (3) 更新G：使用3.2的方法
        G = Y * diag(sum(Y,1).^(-.5));
        
        % (4) 更新u：简化的阈值化方法（避免复杂的连续优化）
        trace_vec = compute_trace_simplified_v2(X, G, Y, nbrIdx, H, params.gamma);
        u = double(trace_vec < tau);
        u = max(u, 0.01);
        
        % 计算目标函数值
        fval = sum(u .* trace_vec) - tau * sum(u);
        
        % 记录
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) median(u) max(u)];
        
        % 更新tau
        tau = tau * 1.01;
    end
    
    disp(result);
end

function trace_vec = compute_trace_simplified_v2(X, G, Y, nbrIdx, H, gamma)
    % 简化的trace计算，使用全局归一化
    n = size(X, 1);
    trace_vec = zeros(n, 1);
    
    % 全局归一化（数值友好近似）
    global_normalizer = diag(sum(Y,1).^(-0.5) + 1e-6);
    
    for i = 1:n
        idx = nbrIdx(i, :);
        Xi = X(idx, :);
        Ki = Xi * Xi';
        
        if rcond(Ki + gamma * eye(size(Ki))) < 1e-12
            Ai = pinv(Ki + gamma * eye(size(Ki)));
        else
            Ai = inv(Ki + gamma * eye(size(Ki)));
        end
        
        % 简化的trace计算
        term = G(idx, :)' * H * Ai * H * Y(idx, :) * global_normalizer;
        trace_vec(i) = trace(term);
    end
end
