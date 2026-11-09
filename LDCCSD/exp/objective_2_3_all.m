function [labels_pred,history] = objective_2_3_all(X, Y_true, params)
    % Algorithm 2.3: Self-paced quadratic form in Y only
    
    n = size(X, 1);                    % 样本数
    c = params.nCluster;               % 聚类类别数
    k = params.k;                      % k近邻参数
    H = eye(k) - (1/k)*ones(k);        % 中心化矩阵
    tau = params.tau;                  % 初始 SPL 门槛 
    taudelt = params.taudelt;          % 每轮递增系数
    tau_max = params.tau_max;
    inner_tol = 1e-5;                  % 内层循环收敛阈值
    max_inner_iter = 30;               % 内层循环最大迭代次数
    nbrIdx = build_knn_graph(X, k);
    
    % ==== 初始化 ====
    u = zeros(n, 1); 
    L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    % 谱初始化聚类
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',1);
    idx2 = randi(n, ceil(0.5*n));
    labels_pred(idx2) = randi(c, ceil(0.5*n));
    Y = full(ind2vec(labels_pred')');
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3f1 init: Acc=', num2str(result_init.Accuracy)]);
    
    fprintf('init compute_trace\n');
    trace_vec_init = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    idx = trace_vec_init < tau;                 
    u(idx) = 1 - (1/tau) * trace_vec_init(idx); 
    
    % 历史记录
    history.obj = [];
    history.tau = [];
    history.u_stat = [];
    history.Accuracy = [];
    history.inner_iters = [];
    
    % 为每个tau阶段单独记录
    history.tau_stages = [];  % 存储每个tau阶段的详细信息
    
    outer_iter = 0;
    total_iter = 0;
    
    % 创建图形窗口
    fig1 = figure('Position', [100, 100, 1200, 400]);
    fig2 = figure('Position', [100, 600, 1200, 400]);
    
    % 外层循环：更新tau
    while tau < tau_max && outer_iter < params.maxIter
        outer_iter = outer_iter + 1;
        fprintf('\n=== Outer iteration %d, tau = %.6f ===\n', outer_iter, tau);
        
        % 为当前tau阶段初始化记录
        current_stage.tau = tau;
        current_stage.obj_values = [];
        current_stage.u_stats = [];
        current_stage.iterations = [];
        
        % 内层循环：固定tau，交替优化Y和u
        inner_iter = 0;
        prev_fval = inf;
        
        while inner_iter < max_inner_iter
            inner_iter = inner_iter + 1;
            total_iter = total_iter + 1;
            
            % Step 1: 固定u，更新Y
            L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma); 
            [Y_hat, obj_F] = solve_Y(L_u, Y);
            alpha = 1;  
            Y = (1 - alpha)*Y + alpha*Y_hat;
            
            % Step 2: 固定Y，更新u
            trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
            u_new = zeros(n, 1);
            idx = trace_vec < tau;                 
            u_new(idx) = 1 - (1/tau) * trace_vec(idx); 
            u_new = max(0, u_new);  
            u = u_new;
            
            % 计算目标函数值
            fval = sum(u .* trace_vec) + ((0.5 * (u.'*u) - sum(u))*tau);
            
            % 评估结果
            [~, labels_pred] = max(Y, [], 2);
            result = my_eval_y(labels_pred, Y_true);
            
            % 记录当前阶段的数据
            current_stage.obj_values = [current_stage.obj_values; fval];
            current_stage.u_stats = [current_stage.u_stats; [min(u), mean(u), max(u)]];
            current_stage.iterations = [current_stage.iterations; inner_iter];
            
            % 记录总历史
            history.obj = [history.obj; fval];
            history.tau = [history.tau; tau];
            history.u_stat = [history.u_stat; [min(u) mean(u) max(u)]];
            history.Accuracy = [history.Accuracy; result.Accuracy];
            
            fprintf('Inner iter %d: Acc=%.4f, fval=%.6f, u_active=%d/%d\n', ...
                    inner_iter, result.Accuracy, fval, sum(u>0), n);
            
            % 检查内层收敛条件
            if abs(prev_fval - fval) < inner_tol
                fprintf('Inner loop converged after %d iterations (delta=%.2e)\n', ...
                        inner_iter, abs(prev_fval - fval));
                break;
            end
            
            prev_fval = fval;
        end
        
        current_stage.final_iter = inner_iter;
        history.tau_stages = [history.tau_stages; current_stage];
        history.inner_iters = [history.inner_iters; inner_iter];
        
        % 绘制当前tau阶段的图
        %plot_tau_stage_results(fig1, fig2, current_stage, outer_iter);
        
        % 更新tau
        if tau < tau_max
            tau = tau * taudelt;
        end
        
        fprintf('Outer iter %d completed: Final Acc=%.4f, Active samples=%d/%d\n', ...
                outer_iter, result.Accuracy, sum(u>0), n);
    end
    
    counts = histcounts(labels_pred, 1:c+1);
    fprintf('\n最终每个簇的样本数:\n');
    disp(counts);
    fprintf('总迭代次数: %d (外层: %d)\n', total_iter, outer_iter);
    fprintf('最终精度: %.4f\n', result.Accuracy);
    
    % 转换历史记录为固定长度数组（为了兼容性）
    max_len = params.maxIter;
    if length(history.obj) < max_len
        history.obj = [history.obj; zeros(max_len - length(history.obj), 1)];
        history.tau = [history.tau; zeros(max_len - length(history.tau), 1)];
        history.u_stat = [history.u_stat; zeros(max_len - size(history.u_stat,1), 3)];
        history.Accuracy = [history.Accuracy; zeros(max_len - length(history.Accuracy), 1)];
    end
    
    % 绘制总体u变化图
    plot_overall_u_statistics(history, outer_iter);
end

function plot_overall_u_statistics(history, num_outer_iters)
    % 绘制整个训练过程中u的总体变化
    
    % 找到有效数据的长度
    valid_len = find(history.obj == 0, 1) - 1;
    if isempty(valid_len)
        valid_len = length(history.obj);
    end
    
    % 创建新的图形窗口
    fig_overall = figure('Position', [200, 200, 1200, 600]);
    
    % 提取有效数据
    iterations = 1:valid_len;
    u_min = history.u_stat(1:valid_len, 1);
    u_mean = history.u_stat(1:valid_len, 2);
    u_max = history.u_stat(1:valid_len, 3);
    tau_values = history.tau(1:valid_len);
    obj_values = history.obj(1:valid_len);
    
    % 子图1: u的统计变化
    subplot(2, 2, 1);
    plot(iterations, u_min, 'b-', 'LineWidth', 2); hold on;
    plot(iterations, u_mean, 'g-', 'LineWidth', 2);
    plot(iterations, u_max, 'r-', 'LineWidth', 2);
    xlabel('Total Iterations');
    ylabel('u Statistics');
    title('Overall u Statistics Evolution');
    legend('min(u)', 'mean(u)', 'max(u)', 'Location', 'best');
    grid on;
    
    % 添加tau变化的右侧y轴
    yyaxis right;
    plot(iterations, tau_values, 'k--', 'LineWidth', 1.5);
    ylabel('τ values');
    yyaxis left; % 切回左侧y轴
    
    % 子图2: 目标函数变化
    subplot(2, 2, 2);
    plot(iterations, obj_values, 'k-', 'LineWidth', 2);
    xlabel('Total Iterations');
    ylabel('Objective Value');
    title('Overall Objective Function Evolution');
    grid on;
    
    % 子图3: u均值 vs tau值的关系
    subplot(2, 2, 3);
    scatter(tau_values, u_mean, 50, iterations, 'filled');
    xlabel('τ values');
    ylabel('mean(u)');
    title('mean(u) vs τ Relationship');
    % 修正colorbar的Label设置
    c = colorbar;
    c.Label.String = 'Iteration';
    grid on;
    
    % 子图4: u活跃样本比例
    subplot(2, 2, 4);
    % 计算活跃样本比例（假设u > 0.1为活跃）
    u_active_ratio = u_mean; % 这里用mean(u)作为活跃程度的近似
    plot(iterations, u_active_ratio * 100, 'Color', [0.5, 0, 0.5], 'LineWidth', 2);
    xlabel('Total Iterations');
    ylabel('Active Sample Ratio (%)');
    title('Active Sample Ratio Evolution');
    grid on;
    
    % 添加tau变化的垂直线到所有子图
    tau_change_points = [];
    for i = 2:length(tau_values)
        if abs(tau_values(i) - tau_values(i-1)) > 1e-8
            tau_change_points = [tau_change_points, i];
        end
    end
    
    for i = 1:4
        subplot(2, 2, i);
        hold on;
        y_limits = ylim;
        for cp = tau_change_points
            line([cp cp], y_limits, 'Color', [0.7, 0.7, 0.7], 'LineStyle', '--', 'LineWidth', 1);
        end
        hold off;
    end
    
    sgtitle(sprintf('Overall u Statistics Summary (%d outer iterations)', num_outer_iters));
    
    % 打印统计信息
    fprintf('\n=== Overall u Statistics Summary ===\n');
    fprintf('Final min(u): %.4f\n', u_min(end));
    fprintf('Final mean(u): %.4f\n', u_mean(end));
    fprintf('Final max(u): %.4f\n', u_max(end));
    fprintf('Initial mean(u): %.4f\n', u_mean(1));
    fprintf('u mean change: %.4f -> %.4f (Δ=%.4f)\n', u_mean(1), u_mean(end), u_mean(end) - u_mean(1));
    fprintf('Final τ value: %.4f\n', tau_values(end));
    fprintf('Final objective: %.4f\n', obj_values(end));
    
    drawnow;
end


function plot_tau_stage_results(fig1, fig2, stage, stage_num)
    % 绘制单个tau阶段的结果
    
    % 确定使用哪个图形窗口和subplot位置
    if stage_num <= 6
        % 使用原始图形窗口
        figure(fig1);
        subplot_pos = stage_num;
        current_fig1 = fig1;
        
        figure(fig2);
        current_fig2 = fig2;
    else
        % 需要创建或使用新的图形窗口
        if stage_num == 7
            % 第一次超过6，创建新窗口
            fig1_new = figure('Position', [1400, 100, 1200, 400]);
            fig2_new = figure('Position', [1400, 600, 1200, 400]);
            setappdata(0, 'fig1_new', fig1_new);
            setappdata(0, 'fig2_new', fig2_new);
        end
        
        % 获取新窗口句柄
        fig1_new = getappdata(0, 'fig1_new');
        fig2_new = getappdata(0, 'fig2_new');
        
        figure(fig1_new);
        subplot_pos = stage_num - 6;
        current_fig1 = fig1_new;
        
        figure(fig2_new);
        current_fig2 = fig2_new;
        
        % 确保subplot位置不超过6
        if subplot_pos > 6
            fprintf('Warning: Too many stages for current subplot layout. Stage %d skipped.\n', stage_num);
            return;
        end
    end
    
    % 绘制u的统计信息
    figure(current_fig1);
    subplot(2, 3, subplot_pos);
    plot(stage.iterations, stage.u_stats(:,1), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
    plot(stage.iterations, stage.u_stats(:,2), 'g-s', 'LineWidth', 1.5, 'MarkerSize', 4);
    plot(stage.iterations, stage.u_stats(:,3), 'r-^', 'LineWidth', 1.5, 'MarkerSize', 4);
    xlabel('Inner Iterations');
    ylabel('u Statistics');
    title(sprintf('τ=%.4f (Stage %d)', stage.tau, stage_num));
    legend('min(u)', 'mean(u)', 'max(u)', 'Location', 'best');
    grid on;
    hold off;
    
    % 绘制目标函数值
    figure(current_fig2);
    subplot(2, 3, subplot_pos);
    plot(stage.iterations, stage.obj_values, 'k-o', 'LineWidth', 2, 'MarkerSize', 4);
    xlabel('Inner Iterations');
    ylabel('Objective Value');
    title(sprintf('Objective Function τ=%.4f (Stage %d)', stage.tau, stage_num));
    grid on;
    
    drawnow; % 立即更新图形
end


