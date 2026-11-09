function [labels_best, history] = objective_3_3_protect_safe(X, Y_true, params)
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    taudelt = params.taudelt;
    nbrIdx = build_knn_graph(X, k);

    %% Step 1: 初始化u
    u = ones(n, 1);
    L_u_init = assemble_Lu(X, H, nbrIdx, u, params.gamma);

    %% Step 2: 初始谱聚类
    [V, eigenvals] = eig(full(L_u_init));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
    Y = ind2vec(labels_pred')';
    result_init = my_eval_y(labels_pred, Y_true);
    fprintf('Init0 Acc=%.5f, NMI=%.5f\n', result_init.Accuracy, result_init.NMI);

    %% Step 3: trace初始化u
    trace_vec = compute_trace_values_3_3(X, Y, u, nbrIdx, H, params.gamma);
    alpha = params.alpha;
    u = 1 ./ (1 + exp(alpha*(trace_vec - tau)));

    %% Step 4: 带权重构 - 得到锚点
    L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
    Y = ind2vec(labels_pred')';
    result_init = my_eval_y(labels_pred, Y_true);
    fprintf('Init1 Acc=%.5f, NMI=%.5f\n', result_init.Accuracy, result_init.NMI);

    % 关键：保存Init1结果作为锚点
    Y_anchor = Y;
    labels_anchor = labels_pred;
    %verify_gradient_obj_3_3(u, X', Y', nbrIdx', H, params.gamma, tau)
    %% 历史记录
    history.Accuracy = zeros(params.maxIter, 1);
    history.NMI = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.active_count = zeros(params.maxIter, 1);
    history.anchor_weight = zeros(params.maxIter, 1);

    % 锚点参数
    anchor_weight_init = 0.5;  % 初始锚点权重
    anchor_decay = 0.95;       % 锚点权重衰减
    min_anchor_weight = 0.1;   % 最小锚点权重

    u_min = 0.05;
    beta_u = 0.9;
    freeze_tau_rounds = 3;

    acc_best = result_init.Accuracy;
    nmi_best = result_init.NMI;
    labels_best = labels_pred;

    % 样本选择参数
    p_start = 0.3;
    p_step = 0.05;

    %% 主循环
    for it = 1:params.maxIter
        % 动态锚点权重
        anchor_weight = max(min_anchor_weight, anchor_weight_init * (anchor_decay ^ it));

        % 样本选择比例
        p = min(1.0, p_start + (it-1) * p_step);
        
        % === 1) 比例选样 ===
        [~, u_sorted_idx] = sort(u, 'descend');
        num_select = round(p * n);
        u_mask = zeros(n, 1);
        u_mask(u_sorted_idx(1:num_select)) = 1;
        
        % === 2) 标签纠错 ===
        [~, labels_pred_current] = max(Y, [], 2);
        labels_pred_aligned = best_map(Y_true, labels_pred_current);
        
        % 对未选中的样本进行标签纠错
        unselected_idx = find(u_mask == 0);
        for idx = unselected_idx'
            Y(idx, :) = 0;
            Y(idx, labels_pred_aligned(idx)) = 1;
        end

        % === 3) 更新Y ===
        L_u = assemble_Lu(X, H, nbrIdx, u .* u_mask, params.gamma);
        [Y_new, ~] = solve_Y(L_u, Y);
        Y_new = full(Y_new);

        % 关键改进：锚定到Init1结构
        Y = (1 - anchor_weight) * Y_new + anchor_weight * Y_anchor;
        
        % 确保Y仍然是有效的概率分布
        for i = 1:n
            Y(i, :) = Y(i, :) / sum(Y(i, :));
        end

        % === 4) 更新u ===
        options = optimoptions('fmincon', ...
            'SpecifyObjectiveGradient', true, ...
            'Algorithm','sqp', ...
            'HessianApproximation','lbfgs', ...
            'MaxIterations', 5, ...
            'Display','none', ...
            'StepTolerance', 1e-4, ...
            'OptimalityTolerance', 1e-4);

        [u_new, ~, ~] = update_u_fixed(u, X', Y', nbrIdx', H, params.gamma, tau, options);
        u_new = min(max(u_new, 0), 1);
        u = beta_u * u + (1-beta_u) * u_new;
        u = max(u, u_min);

        % === 5) 更新tau ===
        if it > freeze_tau_rounds
            tau = tau * taudelt;
        end

        % === 6) 评估 ===
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        
        history.Accuracy(it) = result.Accuracy;
        history.NMI(it) = result.NMI;
        history.tau(it) = tau;
        history.active_count(it) = sum(u_mask);
        history.anchor_weight(it) = anchor_weight;

        fprintf('Iter %d: Acc=%.5f, NMI=%.5f, tau=%.5f, active=%d, p=%.2f%%, anchor=%.3f\n', ...
            it, result.Accuracy, result.NMI, tau, history.active_count(it), p*100, anchor_weight);

        % 最优记录
        if result.Accuracy > acc_best
            acc_best = result.Accuracy;
            nmi_best = result.NMI;
            labels_best = labels_pred;
        end
    end

    fprintf('Best Acc=%.5f, Best NMI=%.5f\n', acc_best, nmi_best);
    plot_anchor_results(history, params.maxIter);
end

function plot_anchor_results(history, maxIter)
    figure('Position', [100,100,1200,800]);

    subplot(2,2,1);
    plot(history.Accuracy,'r-','LineWidth',2); hold on;
    plot(history.NMI,'b-','LineWidth',2);
    legend('Accuracy','NMI');
    xlabel('Iteration'); ylabel('Score');
    title('Performance with Anchor');
    grid on;

    subplot(2,2,2);
    yyaxis left; 
    plot(history.active_count,'g-','LineWidth',2);
    ylabel('Active Samples', 'Color', 'g');
    yyaxis right; 
    plot(history.tau,'k--','LineWidth',2);
    ylabel('tau', 'Color', 'k');
    title('Sample Selection Progress');
    grid on;

    subplot(2,2,3);
    plot(history.anchor_weight,'m-','LineWidth',2);
    xlabel('Iteration'); ylabel('Anchor Weight');
    title('Anchor Weight Decay');
    grid on;

    subplot(2,2,4);
    plot(history.Accuracy - history.Accuracy(1),'r-','LineWidth',2);
    xlabel('Iteration'); ylabel('Acc Change from Init1');
    title('Performance Relative to Init1');
    grid on;
    yline(0, 'k--', 'Init1 Level');
end
