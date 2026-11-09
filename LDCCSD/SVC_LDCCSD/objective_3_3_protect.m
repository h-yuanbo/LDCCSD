function [labels_best, history] = objective_3_3_protect(X, Y_true, params)
    
    n = size(X, 1);
    c = params.nCluster;
    k = params.k;
    H = eye(k) - (1/k)*ones(k);
    tau = params.tau;
    taudelt = params.taudelt;
    nbrIdx = build_knn_graph(X, k);

    % ==== Step 1: u 初始 ====
    u_init = ones(n, 1);
    L_u_init = assemble_Lu(X, H, nbrIdx, u_init, params.gamma);
    
    % ==== Step 2: 谱初始化标签 ====
    [V, eigenvals] = eig(full(L_u_init));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);

    % ==== Step 3: 加噪声 == 可选 ====
%     idx2 = randi(n, ceil(0.3*n));
%     labels_pred(idx2) = randi(c, ceil(0.3*n));
    Y = ind2vec(labels_pred')';
    result_init = my_eval_y(labels_pred, Y_true);

    disp(['Algorithm 3.3 init0 Acc=', num2str(result_init.Accuracy), ...
          ', NMI=', num2str(result_init.NMI)]);

    % ==== Step 4: 数据相关的 u 初始化（Sigmoid） ====
    trace_vec_init = compute_trace_values_3_3(X, Y, u_init, nbrIdx, H, params.gamma);
           
            figure; 
            histogram(trace_vec_init, 50);
            title('Initial Loss Distribution');
            xlabel('Loss value'); 
            ylabel('Count');
    alpha = params.alpha;
    u = 1 ./ (1 + exp(alpha*(trace_vec_init - tau)));
    verify_gradient_obj_3_3(u, X, Y, nbrIdx, H, params.gamma, tau);
    % ==== Step 5: 用带权u重构L 并重新初始化 Y ====
    L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
    [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 1);
    Y = ind2vec(labels_pred')'; 
    result_init = my_eval_y(labels_pred, Y_true);

    disp(['Algorithm 3.3 init1 Acc=', num2str(result_init.Accuracy), ...
          ', NMI=', num2str(result_init.NMI)]);
    % ==== 历史记录 ====
    history.obj = zeros(params.maxIter, 1);
    history.Accuracy = zeros(params.maxIter, 1);
    history.NMI = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.tau = zeros(params.maxIter, 1);

    % 保存历史最优
    result = my_eval_y(labels_pred, Y_true);
    acc_best = result.Accuracy;
    nmi_best = result.NMI;
    labels_best = labels_pred;

    % ==== 训练循环 ====

    freeze_tau_rounds = 3;  % 冻结 tau 的轮数
    % 在 loop 开头或 fval 存储的代码前
 

    % 保存 u 分布，如果样本很多可以间隔保存


    for it = 1:params.maxIter
        % --- (1) 更新 Y ---
        L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
        [Y_new, obj_F] = solve_Y(L_u, Y);
        Y = full(Y_new);

        % --- (2) 计算 trace & 更新 u ---
        trace_vec = compute_trace_values_3_3(X, Y, u, nbrIdx, H, params.gamma);
        if it == 1 || mod(it,5) == 0
            figure; 
            histogram(trace_vec, 50);
            title('Initial Loss Distribution');
            xlabel('Loss value'); 
            ylabel('Count');
        end
        options = optimoptions('fmincon', ...
            'SpecifyObjectiveGradient', true, ...
            'Algorithm','sqp', ...
            'HessianApproximation','lbfgs', ...
            'MaxIterations', 5, ...
            'Display','none', ...
            'StepTolerance', 1e-6, ...
            'OptimalityTolerance', 1e-6);
        [u, tau, fval] = update_u_fixed(u, X', Y', nbrIdx', H, params.gamma, tau, options);

        % --- 动量平滑 + 保底 ---
%         u = beta_u * u + (1 - beta_u) * u_new;
%         u = max(u, u_min);

        % --- (3) 更新 tau（延迟增长） ---
        if it > freeze_tau_rounds && tau < params.tau_max
            tau = tau * taudelt;
        end

        % --- (4) 评估 ---
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);

        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
        history.Accuracy(it) = result.Accuracy;
        history.NMI(it) = result.NMI;

        disp(sprintf('Iter %d: Acc=%.5f, NMI=%.5f, tau=%.5f', ...
            it, result.Accuracy, result.NMI, tau));

        % --- (5) 保存最优 ---
        if result.Accuracy > acc_best
            acc_best = result.Accuracy;
            nmi_best = result.NMI;
            labels_best = labels_pred;
            Y_best = Y;
        end
    end

    disp(sprintf('Best Acc=%.5f, Best NMI=%.5f', acc_best, nmi_best));
    plot_history(history, '3.3 Protect');
        % 曲线1：u的 min / mean / max
    figure;
    plot(1:params.maxIter, history.u_stat(:,1), 'r-', 'LineWidth', 1.5); hold on;
    plot(1:params.maxIter, history.u_stat(:,2), 'g-', 'LineWidth', 1.5);
    plot(1:params.maxIter, history.u_stat(:,3), 'b-', 'LineWidth', 1.5);
    xlabel('Iteration'); ylabel('u value');
    legend('min(u)', 'mean(u)', 'max(u)');
    title('Range of u over iterations');
    grid on;



end
