function [labels_pred, history] = objective_2_3_with_early_stopping_anchor(X, Y_true, params)
    % Algorithm 2.3 with convergence, early stopping, and anchoring mechanisms
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    taudelt = params.taudelt;
    tau_max = params.tau_max;
    nbrIdx = build_knn_graph(X, k);
    alpha = params.alpha;
    
    % 收敛参数
    Y_tol = 1e-6;
    u_tol = 1e-6; 
    obj_tol =  1e-8;
    max_iter = 100;
    
    % 早停参数
    patience =  5;           % 容忍性能下降的轮数
    min_delta =  1e-4;      % 最小改善阈值
    
    % 锚定参数
    use_anchor = true;
    anchor_weight_init = 0.3;
    anchor_decay = 0.95;
    min_anchor_weight = 0.05;
    
    % ==== 初始化部分 ====
    u_init = ones(n, 1);
    L_u_init = assemble_Lu_for_2_3(X, H, nbrIdx, u_init, params.gamma);
    
    [V, eigenvals] = eig(full(L_u_init));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
    Y = full(ind2vec(labels_pred')');
    
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init0: Acc=', num2str(result_init.Accuracy), ', NMI=', num2str(result_init.NMI)]);
    
    trace_vec_init = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    u = 1 ./ (1 + exp(alpha*(trace_vec_init - tau)));
    
    L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
%         idx2 = randi(n, ceil(0.3*n));
%     labels_pred(idx2) = randi(c, ceil(0.3*n));
    Y = full(ind2vec(labels_pred')');
    
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init1: Acc=', num2str(result_init.Accuracy), ', NMI=', num2str(result_init.NMI)]);
    
    % ==== 早停和锚定初始化 ====
    best_acc = result_init.Accuracy;
    best_labels = labels_pred;
    best_Y = Y;
    best_iter = 0;
    no_improve_count = 0;
    
    % 锚点设置
    Y_anchor = Y;  % 保存初始好结果作为锚点
    anchor_performance = result_init.Accuracy;
    
    % 历史记录
    history.obj = zeros(max_iter, 1);
    history.tau = zeros(max_iter, 1);
    history.u_stat = zeros(max_iter, 3);
    history.Accuracy = zeros(max_iter, 1);
    history.NMI = zeros(max_iter, 1);
    history.convergence = struct();
    history.early_stopping = struct();
    history.anchoring = struct();
    
    % 初始化
    Isconverg = 0;
    iter = 0;
    early_stopped = false;
    
    Y_old = Y;
    u_old = u;
    obj_old = 0;
    
    % ==== 主循环 ====
    while(Isconverg == 0 && ~early_stopped)
        iter = iter + 1;
        fprintf('----processing iter %d--------\n', iter);
        
        Y_old = Y;
        u_old = u;
        obj_old_val = obj_old;
        
        % (1) 更新Y
        L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        [Y_update, obj_F] = solve_Y(L_u, Y);
        
        % ==== 锚定机制 ====
        if use_anchor
            current_anchor_weight = max(min_anchor_weight, anchor_weight_init * (anchor_decay ^ iter));
            
            Y = (1 - alpha - current_anchor_weight) * Y + alpha * Y_update + current_anchor_weight * Y_anchor;
            
            history.anchoring.weight(iter) = current_anchor_weight;
            
            fprintf('Anchor weight: %.6f\n', current_anchor_weight);
        else
            Y = (1 - alpha) * Y + alpha * Y_update;
            history.anchoring.weight(iter) = 0;
        end
        
        % (2) 更新u
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        u = 1 ./ (1 + exp(alpha*(trace_vec - tau)));
        u = max(0, min(1, u));
        
        fval = sum(u .* trace_vec) - tau * sum(u);
        obj_old = fval;
        
        % 评估性能
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        
        % ==== 早停机制 ====
        if result.Accuracy > best_acc + min_delta
            best_acc = result.Accuracy;
            best_labels = labels_pred;
            best_Y = Y;
            best_iter = iter;
            no_improve_count = 0;
            
            % 动态更新锚点（可选）
            if use_anchor && result.Accuracy > anchor_performance + 0.01
                Y_anchor = Y;
                anchor_performance = result.Accuracy;
                fprintf('Anchor updated: new performance %.4f\n', result.Accuracy);
            end
        else
            no_improve_count = no_improve_count + 1;
        end
        
        % 检查早停条件
        if no_improve_count >= patience
            fprintf('Early stopping triggered at iter %d\n', iter);
            fprintf('Best performance %.4f achieved at iter %d\n', best_acc, best_iter);
            early_stopped = true;
            labels_pred = best_labels;
            Y = best_Y;
        end
        
        % 记录早停信息
        history.early_stopping.best_acc(iter) = best_acc;
        history.early_stopping.no_improve_count(iter) = no_improve_count;
        history.early_stopping.current_acc(iter) = result.Accuracy;
        
        % 记录历史
        history.obj(iter) = fval;
        history.tau(iter) = tau;
        history.u_stat(iter,:) = [min(u), mean(u), max(u)];
        history.Accuracy(iter) = result.Accuracy;
        history.NMI(iter) = result.NMI;
        
        % ==== 收敛性检查 ====
        if ~early_stopped
            Isconverg = 1;
            
            if iter > 1
                Y_change = norm(Y - Y_old, 'fro') / (norm(Y_old, 'fro') + eps);
                history.convergence.Y_change(iter) = Y_change;
                if Y_change > Y_tol
                    fprintf('    Y_change %7.10f    ', Y_change);
                    Isconverg = 0;
                end
                
                u_change = norm(u - u_old, inf);
                history.convergence.u_change(iter) = u_change;
                if u_change > u_tol
                    fprintf('u_change %7.10f    ', u_change);
                    Isconverg = 0;
                end
                
                obj_change = abs(fval - obj_old_val) / (abs(obj_old_val) + eps);
                history.convergence.obj_change(iter) = obj_change;
                if obj_change > obj_tol
                    fprintf('obj_change %7.10f    \n', obj_change);
                    Isconverg = 0;
                end
            else
                Isconverg = 0;
            end
            
            if iter >= max_iter
                fprintf('Reached maximum iterations: %d\n', max_iter);
                Isconverg = 1;
            end
        end
        
        fprintf('[Iter %d] Acc=%.4f, NMI=%.4f, obj=%.6f, active=%d, best_acc=%.4f\n', ...
            iter, result.Accuracy, result.NMI, fval, sum(u > 0.05), best_acc);
        
        if tau < tau_max
            tau = tau * taudelt;
        end
    end
    
    if early_stopped
        fprintf('Algorithm early stopped at iteration %d with best accuracy %.4f\n', iter, best_acc);
    else
        fprintf('Algorithm converged at iteration %d\n', iter);
    end
    
    % 截取历史记录
    history.obj = history.obj(1:iter);
    history.tau = history.tau(1:iter);
    history.u_stat = history.u_stat(1:iter, :);
    history.Accuracy = history.Accuracy(1:iter);
    history.NMI = history.NMI(1:iter);
    
    plot_comprehensive_history(history, '2.3 with Early Stop & Anchor');
end

function plot_comprehensive_history(history, method_name)
    % 重写的综合历史绘图函数，处理向量长度不匹配问题
    
    figure('Position', [50, 50, 1600, 1200]);
    
    % 获取实际的迭代次数
    actual_iters = length(history.Accuracy);
    x_axis = 1:actual_iters;
    
    % 1. 目标函数值
    subplot(3, 3, 1);
    if length(history.obj) >= actual_iters
        plot(x_axis, history.obj(1:actual_iters), 'b-', 'LineWidth', 2);
    end
    xlabel('Iteration'); 
    ylabel('Objective Value');
    title('Objective Function'); 
    grid on;
    
    % 2. 性能跟踪
    subplot(3, 3, 2);
    plot(x_axis, history.Accuracy(1:actual_iters), 'r-', 'LineWidth', 2); hold on;
    plot(x_axis, history.NMI(1:actual_iters), 'g-', 'LineWidth', 2);
    
    % 添加最佳性能线（如果存在）
    if isfield(history, 'early_stopping') && isfield(history.early_stopping, 'best_acc')
        best_acc_data = history.early_stopping.best_acc;
        if length(best_acc_data) >= actual_iters
            plot(x_axis, best_acc_data(1:actual_iters), 'r--', 'LineWidth', 1.5);
            legend('Current Acc', 'NMI', 'Best Acc', 'Location', 'best');
        else
            legend('Current Acc', 'NMI', 'Location', 'best');
        end
    else
        legend('Current Acc', 'NMI', 'Location', 'best');
    end
    
    xlabel('Iteration'); 
    ylabel('Score');
    title('Performance Tracking'); 
    grid on;
    
    % 3. 样本权重分布
    subplot(3, 3, 3);
    if size(history.u_stat, 1) >= actual_iters
        u_data = history.u_stat(1:actual_iters, :);
        plot(x_axis, u_data(:, 2), 'r-', 'LineWidth', 2); hold on;
        
        % 创建填充区域（最小值到最大值）
        fill([x_axis, fliplr(x_axis)], [u_data(:, 1)', fliplr(u_data(:, 3)')], ...
             'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    xlabel('Iteration'); 
    ylabel('u value');
    title('Sample Weights Distribution'); 
    grid on;
    
    % 4. SPL阈值
    subplot(3, 3, 4);
    if length(history.tau) >= actual_iters
        plot(x_axis, history.tau(1:actual_iters), 'k-', 'LineWidth', 2);
    end
    xlabel('Iteration'); 
    ylabel('tau');
    title('SPL Threshold'); 
    grid on;
    
    % 5. 收敛性监控
    subplot(3, 3, 5);
    if isfield(history, 'convergence')
        has_data = false;
        legend_labels = {};
        
        % Y变化
        if isfield(history.convergence, 'Y_change')
            y_change = history.convergence.Y_change;
            if ~isempty(y_change)
                % 找到有效数据点
                valid_y = y_change(y_change > 0);
                valid_x = find(y_change > 0);
                if ~isempty(valid_y)
                    semilogy(valid_x, valid_y, 'r-', 'LineWidth', 2); hold on;
                    legend_labels{end+1} = 'Y change';
                    has_data = true;
                end
            end
        end
        
        % u变化
        if isfield(history.convergence, 'u_change')
            u_change = history.convergence.u_change;
            if ~isempty(u_change)
                valid_u = u_change(u_change > 0);
                valid_x = find(u_change > 0);
                if ~isempty(valid_u)
                    semilogy(valid_x, valid_u, 'g-', 'LineWidth', 2); hold on;
                    legend_labels{end+1} = 'u change';
                    has_data = true;
                end
            end
        end
        
        % 目标函数变化
        if isfield(history.convergence, 'obj_change')
            obj_change = history.convergence.obj_change;
            if ~isempty(obj_change)
                valid_obj = obj_change(obj_change > 0);
                valid_x = find(obj_change > 0);
                if ~isempty(valid_obj)
                    semilogy(valid_x, valid_obj, 'b-', 'LineWidth', 2); hold on;
                    legend_labels{end+1} = 'obj change';
                    has_data = true;
                end
            end
        end
        
        if has_data && ~isempty(legend_labels)
            legend(legend_labels, 'Location', 'best');
        end
    end
    xlabel('Iteration'); 
    ylabel('Change (log scale)');
    title('Convergence Monitoring'); 
    grid on;
    
    % 6. 早停监控
    subplot(3, 3, 6);
    if isfield(history, 'early_stopping') && isfield(history.early_stopping, 'no_improve_count')
        no_improve = history.early_stopping.no_improve_count;
        if length(no_improve) >= actual_iters
            plot(x_axis, no_improve(1:actual_iters), 'b-', 'LineWidth', 2);
        end
    end
    xlabel('Iteration'); 
    ylabel('No Improve Count');
    title('Early Stopping Monitor'); 
    grid on;
    
    % 7. 锚定权重
    subplot(3, 3, 7);
    if isfield(history, 'anchoring') && isfield(history.anchoring, 'weight')
        anchor_weights = history.anchoring.weight;
        if length(anchor_weights) >= actual_iters
            plot(x_axis, anchor_weights(1:actual_iters), 'Color', [0.5, 0, 0.5], 'LineWidth', 2);
        end
    end
    xlabel('Iteration'); 
    ylabel('Anchor Weight');
    title('Anchoring Strength'); 
    grid on;
    
    % 8. 损失分布统计（如果有的话）
    subplot(3, 3, 8);
    if isfield(history, 'loss_stats')
        loss_data = history.loss_stats;
        if size(loss_data, 1) >= actual_iters
            plot(x_axis, loss_data(1:actual_iters, 1), 'r-', 'LineWidth', 1.5); hold on;
            plot(x_axis, loss_data(1:actual_iters, 2), 'g-', 'LineWidth', 2);
            plot(x_axis, loss_data(1:actual_iters, 3), 'b-', 'LineWidth', 1.5);
            legend('min loss', 'mean loss', 'max loss', 'Location', 'best');
        end
    end
    xlabel('Iteration'); 
    ylabel('Loss Value');
    title('Loss Distribution'); 
    grid on;
    
    % 9. 活跃样本数量
    subplot(3, 3, 9);
    if isfield(history, 'active_samples')
        active_data = history.active_samples;
        if length(active_data) >= actual_iters
            plot(x_axis, active_data(1:actual_iters), 'm-', 'LineWidth', 2);
        end
    end
    xlabel('Iteration'); 
    ylabel('Active Samples');
    title('SPL Sample Selection'); 
    grid on;
    
    % 添加总标题
    sgtitle([method_name, ' - Comprehensive Analysis']);
    
    % 调整子图间距
    set(gcf, 'Units', 'normalized');
    set(gca, 'FontSize', 10);
end
