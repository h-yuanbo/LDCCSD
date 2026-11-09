function [labels_pred, history] = objective_2_3_with_33_init(X, Y_true, params)
    % Algorithm 2.3 with 3.3-style initialization
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    taudelt = params.taudelt;
    tau_max = params.tau_max;
    nbrIdx = build_knn_graph(X, k);
    alpha = params.alpha;
    
    % ==== Step 1: 用全1权重的初始u构造初始L ====
    u = zeros(n, 1);
    L_u_init = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    
    % ==== Step 2: 初始谱聚类 ====
    [V, eigenvals] = eig(full(L_u_init));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
         idx2 = randi(n, ceil(0.3*n));
     labels_pred(idx2) = randi(c, ceil(0.3*n));
    Y = full(ind2vec(labels_pred')');
    
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init0: Acc=', num2str(result_init.Accuracy), ', NMI=', num2str(result_init.NMI)]);
    
    % ==== Step 3: 计算一次损失分布 ====
    trace_vec_init = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    
    figure;
    histogram(trace_vec_init, 50);
    xlabel('Loss value');
    ylabel('Count');
    title('Initial Loss Distribution (2.3)');
    
    % ==== 基于损失的自适应u初始化 ====
    % 使用sigmoid函数而不是硬阈值
     %u = 1 ./ (1 + exp(alpha*(trace_vec_init - tau)));
     u = double(trace_vec_init < tau);
%      L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
%      [V, eigenvals] = eig(full(L_u));
%      [~, idx] = sort(diag(eigenvals), 'ascend');
%      G = real(V(:, idx(2:c+1)));
%      [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
%      idx2 = randi(n, ceil(0.3*n));
%      labels_pred(idx2) = randi(c, ceil(0.3*n));
%      Y = full(ind2vec(labels_pred')');
%      trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
%      figure;
%             histogram(trace_vec, 50);
%             title(sprintf('Iter2 - Loss Distribution'));
%             xlabel('Loss value');
%             ylabel('Count');
%     result_init = my_eval_y(labels_pred, Y_true);
%     disp(['Algorithm 2.3 init1: Acc=', num2str(result_init.Accuracy), ', NMI=', num2str(result_init.NMI)]);
    
    % 历史记录
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.Accuracy = zeros(params.maxIter, 1);
    history.NMI = zeros(params.maxIter, 1);
    
    tol = 1e-12;
    
    % ==== 主循环 ====
    for it = 1:params.maxIter
        % (1) 更新Y
        L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        [Y, obj_F] = solve_Y(L_u, Y);
        %Y = (1 - alpha) * Y + alpha * Y_update;       
        % (2) 更新u - 基于当前Y计算新的损失
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        if it == 1 || mod(it, 5) == 0
            figure;
            histogram(trace_vec, 50);
            title(sprintf('Iter %d - Loss Distribution', it));
            xlabel('Loss value');
            ylabel('Count');
        end
%         
        % SPL样本选择 - 可以选择硬阈值或软阈值
%         if strcmp(params.spl_mode, 'soft')
%             % 软选择：使用sigmoid
%             u = 1 ./ (1 + exp(alpha*(trace_vec - tau)));
%         else
%             % 硬选择：阈值化
           u = double(trace_vec < tau);
%         end
        
        u = max(0, min(1, u));
        
        % 计算目标函数值
        fval = sum(u .* trace_vec) - tau * sum(u);
        
        % 评估性能
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        
        disp(['Algorithm 2.3 iter', num2str(it), ': Acc=', num2str(result.Accuracy), ...
              ', NMI=', num2str(result.NMI), ', obj=', num2str(fval), ...
              ', active=', num2str(sum(u > 0.05))]);
        
        % 记录历史
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u), mean(u), max(u)];
        history.Accuracy(it) = result.Accuracy;
        history.NMI(it) = result.NMI;
        
        % 更新tau
        if tau < tau_max
            tau = tau * taudelt;
        end
        
    end
    
    % 最终统计
    counts = histcounts(labels_pred, 1:c+1);
    fprintf('最终每个簇的样本数:\n');
    disp(counts);
    
    plot_history_2_3(history, '2.3 with 3.3 init');
end

function plot_history_2_3(history, method_name)
    figure('Position', [100, 100, 1200, 800]);
    
    subplot(2, 2, 1);
    plot(history.obj, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Objective Value');
    title([method_name, ' - Objective']);
    grid on;
    
    subplot(2, 2, 2);
    plot(history.Accuracy, 'r-', 'LineWidth', 2); hold on;
    plot(history.NMI, 'g-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Score');
    legend('Accuracy', 'NMI');
    title([method_name, ' - Performance']);
    grid on;
    
    subplot(2, 2, 3);
    plot(history.u_stat(:,1), 'b--', 'LineWidth', 1.5); hold on;
    plot(history.u_stat(:,2), 'r-', 'LineWidth', 2);
    plot(history.u_stat(:,3), 'b--', 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('u value');
    legend('min(u)', 'mean(u)', 'max(u)');
    title([method_name, ' - Sample Weights']);
    grid on;
    
    subplot(2, 2, 4);
    plot(history.tau, 'k-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('tau');
    title([method_name, ' - SPL Threshold']);
    grid on;
end
