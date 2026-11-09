function [labels_pred, history] = objective_1_3_linear_soft(X, Y_true, params)
    % Algorithm 1.3: Linear soft regularized self-paced clustering
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    lambda = params.tau;  % 软正则化参数
    lambda_growth = params.taudelt;  % λ增长率
    nbrIdx = build_knn_graph(X, k);
    
    % 初始化v
    v = ones(n, 1);  % 开始时所有样本权重为1
    
    % 初始化Y
    L_v = assemble_Lu_soft(X, H, nbrIdx, v, params.gamma);
    [V, eigenvals] = eig(full(L_v));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50);
    Y = ind2vec(labels_pred')';
    
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 1.3 Linear Soft init: Acc=', num2str(result_init.Accuracy)]);
    
    history.obj = zeros(params.maxIter, 1);
    history.lambda = zeros(params.maxIter, 1);
    history.v_stat = zeros(params.maxIter, 3);
    
    for it = 1:params.maxIter
        % (1) 更新Y - 固定v
        L_v = assemble_Lu_soft(X, H, nbrIdx, v, params.gamma);
        [Y, obj_F] = solve_Y(L_v, Y);
        [~, labels_pred] = max(Y, [], 2);
        
        % (2) 更新v - 使用闭式解
        trace_vec = compute_trace_values_1_3(X, Y, nbrIdx, H, params.gamma);
        v = update_v_linear_soft(trace_vec, lambda);
        
        % 计算目标函数值
        fval = sum(v .* trace_vec) - lambda * sum(v);
        
        % 记录和显示
        result = my_eval_y(labels_pred, Y_true);
        disp(['Iter ', num2str(it), ': Acc=', num2str(result.Accuracy), ...
              ', Obj=', num2str(fval), ', λ=', num2str(lambda), ...
              ', mean(v)=', num2str(mean(v))]);
        
        history.obj(it) = fval;
        history.lambda(it) = lambda;
        history.v_stat(it,:) = [min(v) mean(v) max(v)];
        
        % 更新λ（课程学习：逐渐变难）
        lambda = lambda * lambda_growth;
    end
    
    [~, labels_pred] = max(Y, [], 2);
end

function v = update_v_linear_soft(L, lambda)
    % 线性软正则化的闭式解
    % v_i* = max(0, 1 - L_i/λ)
    
    v = max(0, 1 - L / lambda);
    
    % 也可以加上上界约束
    v = min(v, 1);
end

function L_v = assemble_Lu_soft(X, H, nbrIdx, v, gamma)
    % 软权重版本的拉普拉斯矩阵组装
    n = size(X,1);
    k = size(H,1);
    L_v = zeros(n,n);
    
    for i = 1:n
        if v(i) > 1e-6  % 软阈值
            Ni = nbrIdx(i,:);
            Xi = X(Ni,:);
            Ki = Xi' * Xi;
            Mi = Ki + gamma * eye(size(Ki,1));
            Li = H * (Xi * (Mi \ Xi')) * H;
            L_v(Ni,Ni) = L_v(Ni,Ni) + v(i) * Li;  % v(i)连续权重
        end
    end
    L_v = 0.5*(L_v + L_v');
end

function trace_vec = compute_trace_values_1_3(X, Y, nbrIdx, H, gamma)
    % 计算每个样本的trace损失
    n = size(X, 1);
    trace_vec = zeros(n, 1);
    
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);        % k×d
        Yi = Y(Ni,:);        % k×c
        
        % L^i = H^i (X^{iT}X^i + γI)^{-1} H^i
        Ki = Xi' * Xi;       % d×d
        Mi = Ki + gamma * eye(size(Ki,1));
        Li = H * (Xi * (Mi \ Xi')) * H;  % k×k
        
        % 双重Y归一化
        YtY = Yi' * Yi;
        [U, S, V] = svd(YtY + eps*eye(size(YtY)));
        invsqrt_YtY = U * diag(1./sqrt(diag(S))) * V';
        
        temp = Yi' * Li * Yi;  % c×c
        trace_vec(i) = trace(invsqrt_YtY * temp * invsqrt_YtY);
    end
end
