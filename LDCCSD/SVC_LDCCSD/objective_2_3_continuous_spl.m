function [labels_pred,history] = objective_2_3_continuous_spl(X, Y_true, params)
    % Algorithm 2.3: Self-paced quadratic form in Y only
    % Modified to use continuous self-paced weights (sigmoid form) for u update.

    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    taudelt = params.taudelt; % This will now be the growth factor for lambda
    nbrIdx = build_knn_graph(X, k);
    
    % 从 params 中获取初始的 lambda 值。
    % 请确保 params 包含 params.lambda_initial 参数。
    initial_lambda = params.tau; 

    % 初始化 u。最初，所有样本都被考虑，因此 u=ones。
    u = ones(n, 1); 
    
    % Initial L_u and Y for initial clustering result
    % 注意: assemble_Lu_for_2_3 在这里使用 u=ones(n,1)，因此它实际上是一个完整的图拉普拉斯矩阵
    L_u_initial = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u_initial));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1))); % 选择最小的 c 个特征值对应的特征向量（跳过第一个，通常是0）
    
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',100);
    % 重要：请检查并注释掉模拟噪声的代码行，除非您明确打算在实验中引入噪声
    idx2 = randi(n, ceil(0.3*n));
    labels_pred(idx2) = randi(c, ceil(0.3*n));

    Y = ind2vec(labels_pred')';

    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init (Continuous SPL): Acc=', num2str(result_init.Accuracy)]);

    history.obj = zeros(params.maxIter, 1);
    history.lambda = zeros(params.maxIter, 1); % 记录 lambda 值 (取代 tau)
    history.u_stat = zeros(params.maxIter, 3); % 记录 u 的最小值、均值、最大值
    history.u_count = zeros(params.maxIter, 2); % 用于可视化 (计算 u < 0.5 和 u >= 0.5 的数量)

%     figure;
%     h1 = plot(NaN,NaN,'r-','LineWidth',1.5); hold on;
%     h2 = plot(NaN,NaN,'b-','LineWidth',1.5);
%     xlabel('Iteration');
%     ylabel('Count');
%     legend({'#u < 0.5','#u >= 0.5'}, 'Location','best'); % 更新图例
%     title('Dynamic change of u counts (Continuous SPL)');
%     grid on;

    fval=0;
    current_lambda = initial_lambda; % 自步参数 lambda，它会在迭代中增长
    
    for it = 1:params.maxIter
        % (1) 更新Y - 主要优化变量
        % L_u 使用当前的连续 u 权重构建
        L_u_current = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        [Y, obj_F] = solve_Y(L_u_current, Y);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        
        % (2) 计算原始的 trace_vec (样本损失 l_vi)
        % 这是公式 (7) 中的单个损失项 l_vi
        raw_trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        
        % (3) 更新 u (使用 sigmoid 形式, 公式 8)
        % 防止 current_lambda 极小时出现除零错误，尽管它应该会增长
        lambda_inv = 1 / (current_lambda + eps); % 添加 eps 以提高数值稳定性
        
        % 使用 sigmoid 公式 w_vi^* = (1 + e^(-1/λ)) / (1 + e^(l_vi - 1/λ)) 计算 u_i
        % MATLAB 的 ./ 运算符用于逐元素除法，exp() 用于逐元素指数
        u = (1 + exp(-lambda_inv)) ./ (1 + exp(raw_trace_vec - lambda_inv));
        
        % (4) 计算目标函数 fval
        % 采用通用形式：sum_i u_i * l_i - lambda * sum_i u_i
        % 注意：如果使用了导致 sigmoid 的特定 f(w_vi) 函数，这个 fval 形式可能会改变。
        % 但对于典型的自步正则化，这是常用的目标函数监测形式。
        fval = sum(u .* raw_trace_vec) - current_lambda * sum(u);
        
        disp(['Algorithm 2.3 iter ', num2str(it), ': Acc=', num2str(result.Accuracy), ...
              '    fval=', num2str(fval), ...
              '    lambda=', num2str(current_lambda), ...
              '    mean_u=', num2str(mean(u))]);
        
        % 记录历史数据
        history.obj(it) = fval;
        history.lambda(it) = current_lambda;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];

        % (5) lambda 的增长策略
        % 在连续自步学习中，lambda 通常会持续增长。
        % 离散自步学习中的停止条件 `mean(u) < 1` 在这里不太适用，
        % 因为 u 值是连续的，可能接近 1 但很少精确达到 1。
        % 我们可以让 lambda 在所有 params.maxIter 中都增长，或者在 mean(u) 接近 1 时停止。
        % 这里使用一个软停止条件，当平均权重非常接近 1 时，停止 lambda 的增长。
        if mean(u) < (1 - 1e-6) % 如果平均权重尚未非常接近 1
            current_lambda = current_lambda * taudelt;
        else
            % fprintf('  INFO: Mean(u) is very close to 1. Stopping lambda growth.\n');
        end
        
        % 为了可视化，计算权重低于/高于某个阈值（例如 0.5）的样本数量
%         history.u_count(it,1) = sum(u < 0.5); 
%         history.u_count(it,2) = sum(u >= 0.5); 
%         set(h1,'XData',1:it,'YData',history.u_count(1:it,1));
%         set(h2,'XData',1:it,'YData',history.u_count(1:it,2));
%         drawnow;
    end
    
    disp(result);
    % 如果您有 plot_history 函数，可以取消注释以下行
     %plot_history(history, '2.3_continuous_spl'); 
end
