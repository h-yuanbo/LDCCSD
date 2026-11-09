function [labels_pred, history] = objective_2_3_with_convergence(X, Y_true, params)
    % Algorithm 2.3 with tensor-inspired convergence conditions
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    taudelt = params.taudelt;
    tau_max = params.tau_max;
    nbrIdx = build_knn_graph(X, k);
    alpha = params.alpha;
    
    % 收敛参数设置
    Y_tol = 1e-6;      % Y矩阵收敛容差
    u_tol =1e-6;      % 权重u收敛容差  
    obj_tol = 1e-8;  % 目标函数收敛容差
    max_iter = 100;
    
    % ==== 初始化部分保持不变 ====
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
    Y = full(ind2vec(labels_pred')');
    
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3 init1: Acc=', num2str(result_init.Accuracy), ', NMI=', num2str(result_init.NMI)]);
    
    % 历史记录
    history.obj = zeros(max_iter, 1);
    history.tau = zeros(max_iter, 1);
    history.u_stat = zeros(max_iter, 3);
    history.Accuracy = zeros(max_iter, 1);
    history.NMI = zeros(max_iter, 1);
    history.convergence = struct();
    
    % 收敛标志
    Isconverg = 0;
    iter = 0;
    
    % 初始化前一轮的值
    Y_old = Y;
    u_old = u;
    obj_old = 0;
    
    % ==== 主循环 ====
    while(Isconverg == 0)
        iter = iter + 1;
        fprintf('----processing iter %d--------\n', iter);
        
        % 保存上一轮的值
        Y_old = Y;
        u_old = u;
        obj_old_val = obj_old;
        
        % (1) 更新Y
        L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        [Y_update, obj_F] = solve_Y(L_u, Y);
        Y = (1 - alpha) * Y + alpha * Y_update;
        
        % (2) 更新u
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        u = 1 ./ (1 + exp(alpha*(trace_vec - tau)));
        u = max(0, min(1, u));
        
        % 计算目标函数值
        fval = sum(u .* trace_vec) - tau * sum(u);
        obj_old = fval;
        
        % 评估性能
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        
        % 记录历史
        history.obj(iter) = fval;
        history.tau(iter) = tau;
        history.u_stat(iter,:) = [min(u), mean(u), max(u)];
        history.Accuracy(iter) = result.Accuracy;
        history.NMI(iter) = result.NMI;
        
        % ==== 收敛性检查 (仿照张量方法) ====
        Isconverg = 1;  % 假设收敛
        
        if iter > 1
            % 1. Y矩阵变化检查
            Y_change = norm(Y - Y_old, 'fro') / (norm(Y_old, 'fro') + eps);
            history.convergence.Y_change(iter) = Y_change;
            if Y_change > Y_tol
                fprintf('    Y_change %7.10f    ', Y_change);
                Isconverg = 0;
            end
            
            % 2. 权重u变化检查  
            u_change = norm(u - u_old, inf);
            history.convergence.u_change(iter) = u_change;
            if u_change > u_tol
                fprintf('u_change %7.10f    ', u_change);
                Isconverg = 0;
            end
            
            % 3. 目标函数变化检查
            obj_change = abs(fval - obj_old_val) / (abs(obj_old_val) + eps);
            history.convergence.obj_change(iter) = obj_change;
            if obj_change > obj_tol
                fprintf('obj_change %7.10f    \n', obj_change);
                Isconverg = 0;
            end
        else
            Isconverg = 0;  % 第一轮不检查收敛
        end
        
        % 最大迭代限制
        if iter >= max_iter
            fprintf('Reached maximum iterations: %d\n', max_iter);
            Isconverg = 1;
        end
        
        % 输出当前状态
        fprintf('[Iter %d] loss: min=%.6f, max=%.6f, std=%.6f\n', ...
            iter, min(trace_vec), max(trace_vec), std(trace_vec));
        
        disp(['Algorithm 2.3 iter', num2str(iter), ': Acc=', num2str(result.Accuracy), ...
              ', NMI=', num2str(result.NMI), ', obj=', num2str(fval), ...
              ', active=', num2str(sum(u > 0.05))]);
        
        % 更新tau
        if tau < tau_max
            tau = tau * taudelt;
        end
    end
    
    fprintf('Algorithm converged at iteration %d\n', iter);
    
    % 截取实际使用的历史记录
    history.obj = history.obj(1:iter);
    history.tau = history.tau(1:iter);
    history.u_stat = history.u_stat(1:iter, :);
    history.Accuracy = history.Accuracy(1:iter);
    history.NMI = history.NMI(1:iter);
    
    % 最终统计
    counts = histcounts(labels_pred, 1:c+1);
    fprintf('最终每个簇的样本数:\n');
    disp(counts);
    
    plot_history_with_convergence(history, '2.3 with convergence');
end

function plot_history_with_convergence(history, method_name)
    figure('Position', [100, 100, 1500, 1000]);
    
    subplot(3, 2, 1);
    plot(history.obj, 'b-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Objective Value');
    title([method_name, ' - Objective']);
    grid on;
    
    subplot(3, 2, 2);
    plot(history.Accuracy, 'r-', 'LineWidth', 2); hold on;
    plot(history.NMI, 'g-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('Score');
    legend('Accuracy', 'NMI');
    title([method_name, ' - Performance']);
    grid on;
    
    subplot(3, 2, 3);
    plot(history.u_stat(:,1), 'b--', 'LineWidth', 1.5); hold on;
    plot(history.u_stat(:,2), 'r-', 'LineWidth', 2);
    plot(history.u_stat(:,3), 'b--', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('u value');
    legend('min(u)', 'mean(u)', 'max(u)');
    title([method_name, ' - Sample Weights']);
    grid on;
    
    subplot(3, 2, 4);
    plot(history.tau, 'k-', 'LineWidth', 2);
    xlabel('Iteration'); ylabel('tau');
    title([method_name, ' - SPL Threshold']);
    grid on;
    
    % 收敛性图
    if isfield(history, 'convergence')
        subplot(3, 2, 5);
        if isfield(history.convergence, 'Y_change')
            semilogy(history.convergence.Y_change, 'r-', 'LineWidth', 2); hold on;
        end
        if isfield(history.convergence, 'u_change')
            semilogy(history.convergence.u_change, 'g-', 'LineWidth', 2); hold on;
        end
        if isfield(history.convergence, 'obj_change')
            semilogy(history.convergence.obj_change, 'b-', 'LineWidth', 2);
        end
        xlabel('Iteration'); ylabel('Change (log scale)');
        legend('Y change', 'u change', 'obj change');
        title('Convergence Monitoring');
        grid on;
    end
end
