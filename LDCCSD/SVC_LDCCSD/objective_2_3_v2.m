function [labels_pred,history] = objective_2_3_v2(X, Y_true, params)
    % Algorithm 2.3: Self-paced quadratic form in Y only
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    taudelt=params.taudelt;
    nbrIdx = build_knn_graph(X, k);
    initial_selection_percentage=params.initial_selection_percentage; % 从params中获取

    % 初始化
    u =ones(n, 1);
    
    % Initial L_u and Y for initial clustering result
    % Note: L_u uses u=ones(n,1) here, so it's effectively a full graph Laplacian
    L_u_initial = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u_initial)); % Use L_u_initial
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',100);
    idx2 = randi(n, ceil(0.3*n)); % If simulating noise is intended
    labels_pred(idx2) = randi(c, ceil(0.3*n)); % If simulating noise is intended

    Y = ind2vec(labels_pred')';
    trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init: Acc=', num2str(result_init.Accuracy)]);

    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.u_count = zeros(params.maxIter, 2);
% 
%     figure;
%     h1 = plot(NaN,NaN,'r-','LineWidth',1.5); hold on;
%     h2 = plot(NaN,NaN,'b-','LineWidth',1.5);
%     xlabel('Iteration');
%     ylabel('Count');
%     legend({'#u=0','#u=1'}, 'Location','best');
%     title('Dynamic change of u counts in Algorithm 2.3');
%     grid on;

    fval=0;
    tau_scaled = 0; % 用于内部计算的 scaled tau
    scaling_factor = 1; % 用于 scaling trace_vec
    
    for it = 1:params.maxIter
        % (1) 更新Y - 主要优化变量
        L_u_current = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma); % 使用当前的u构建L_u
        [Y, obj_F] = solve_Y(L_u_current, Y);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        
        % (2) 计算原始的 trace_vec (样本损失)
        raw_trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        
        % (3) 在第一次迭代时，初始化 tau_scaled 并确定 scaling_factor
        if it == 1
            % 确定 scaling_factor。可以根据均值、中位数或最大值来缩放。
            % 均值：使 trace_vec 的均值变为 1。
            if mean(raw_trace_vec) ~= 0
                scaling_factor = 1 / mean(raw_trace_vec); 
            else
                % 如果所有损失都为0，则无法缩放，可能需要检查数据或模型
                scaling_factor = 1; % 保持不变
            end
            
            % 使用缩放后的 trace_vec 初始化 tau_scaled
            scaled_trace_vec = raw_trace_vec * scaling_factor;
            tau_scaled = initialize_tau_from_loss(scaled_trace_vec, initial_selection_percentage);
            
            disp(['Min raw trace_vec: ', num2str(min(raw_trace_vec))]);
            disp(['Max raw trace_vec: ', num2str(max(raw_trace_vec))]);
            disp(['Mean raw trace_vec: ', num2str(mean(raw_trace_vec))]);
            disp(['Median raw trace_vec: ', num2str(median(raw_trace_vec))]);
            disp(['Initial tau (scaled): ', num2str(tau_scaled)]); % 这是用于比较的缩放后的tau
            disp(['Initial tau (raw threshold): ', num2str(tau_scaled / scaling_factor)]); % 对应的原始损失阈值
        end
        
        % (4) 使用缩放后的 trace_vec 更新 u
        % 在后续迭代中，应用第一次迭代确定的相同 scaling_factor
        current_scaled_trace_vec = raw_trace_vec * scaling_factor;
        u = double(current_scaled_trace_vec < tau_scaled); % 用缩放后的损失与缩放后的tau比较
        
        % (5) 计算目标函数 fval (使用原始的 trace_vec 和原始的 tau 阈值)
        % tau_scaled / scaling_factor 恢复到原始的损失值尺度
        fval = sum(u .* raw_trace_vec) - (tau_scaled / scaling_factor) * sum(u);
        
        disp(['Algorithm 2.3 iter ', num2str(it), ': Acc=', num2str(result.Accuracy), ...
              '    fval=', num2str(fval), '    tau_scaled=', num2str(tau_scaled), ...
              '    tau_raw_threshold=', num2str(tau_scaled / scaling_factor), ...
              '    num_selected_u=', num2str(sum(u))]);
        
        % 记录
        history.obj(it) = fval;
        history.tau(it) = tau_scaled / scaling_factor; % 记录对应的原始tau阈值
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
        history.u_count(it,1) = sum(u==0);
        history.u_count(it,2) = sum(u==1);

        % (6) tau 的增长策略 (应用于缩放后的 tau)
        if mean(u) < 1  % 当不是所有样本都被选中时，继续增长tau
            tau_scaled = tau_scaled * taudelt;
        end
        
%         set(h1,'XData',1:it,'YData',history.u_count(1:it,1));
%         set(h2,'XData',1:it,'YData',history.u_count(1:it,2));
%         drawnow;
    end
    
%     disp(result);
%     plot_history(history, '2.3'); % 如果有此函数
end
