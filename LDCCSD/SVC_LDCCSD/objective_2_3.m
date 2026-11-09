function [labels_pred,history] = objective_2_3(X, Y_true, params)
    % Algorithm 2.3: Self-paced quadratic form in Y only
    % 对应目标函数2.3: min_{u,Y} sum_i u_i * tr[(Y^{iT}Y^i)^{-1/2} Y^{iT} H^i (...) H^i Y^i (Y^{iT}Y^i)^{-1/2}] - τ sum_i u_i
    
    %s_rng = rng; rng(1,'twister');

    n = size(X, 1);                    % 样本数
    c = params.nCluster;               % 聚类类别数
    k = params.k;                       % k近邻参数
    H = eye(k) - (1/k)*ones(k);        % 中心化矩阵
    tau = params.tau;            % 初始 SPL 门槛 
    taudelt = params.taudelt;       % 每轮递增系数
    tau_max=params.tau_max;
    nbrIdx = build_knn_graph(X, k);
    u =0.2+0.3*rand(n,1);
    L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    
% 选择最小的c个特征值对应的特征向量（跳过第一个，通常是0）
G = real(V(:, idx(2:c+1)));

[labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates', 100);
idx2 = randi(n, ceil(0.3*n));
labels_pred(idx2) = randi(c, ceil(0.3*n));

Y = ind2vec(labels_pred')';
% ---- 初始化 ----
% lab = randi(c, [n,1]);
% Y = full(sparse((1:n)', lab, 1, n, c));
result = my_eval_y(labels_pred, Y_true);
disp(['Algorithm 2.3 init: Acc=', num2str(result.Accuracy)]);
trace_vec_init = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    figure; 
    histogram(trace_vec_init, 50);
    title(' Loss init Y Distribution');
    xlabel('Loss value'); 
    ylabel('Count');

%       idx = trace_vec_init < tau;                 % 满足条件的样本
%       u(idx) = 1 - (1/tau) * trace_vec_init(idx); % 公式里的 1 - (1/tau)*trace
% ---- 记录器 ----
    history.obj = zeros(params.maxIter, 1);
    history.beta = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.Accuracy=zeros(params.maxIter, 1);




    % ==== 初始化 ====
    
   
%     trace_vec_init = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
%     figure; 
%     histogram(trace_vec_init, 50);
%     title(' Loss init  Distribution');
%     xlabel('Loss value'); 
%     ylabel('Count');
%     tau_init=params.tau_init;
%     idx = trace_vec_init < tau_init;                 % 满足条件的样本
%     u(idx) = 1 - (1/tau_init) * trace_vec_init(idx); % 公式里的 1 - (1/tau)*trace
%     u = double(trace_vec_init <tau_init);
    %u = zeros(n, 1);
    
    % 历史记录


    % 固定初始化，完全确定
        % ==== 初始化绘图窗口 ====
%     fig = figure;
%     subplot(1,2,1);
%     h_trace = histogram(0,50);  % 占位
%     title('Trace distribution');
%     xlabel('Loss value'); ylabel('Count');
% 
%     subplot(1,2,2);
%     h_u = histogram(0,50,'BinLimits',[0 1],'Normalization','probability');  
%     % BinLimits=[0 1] 保证范围固定在 0~1
%     title('u distribution');
%     xlabel('u value'); ylabel('Frequency');
%     ylim([0 1]);  % 固定纵轴范围

    % ==== 迭代 ====
    for it = 1:params.maxIter
        % (1) 更新Y
        L_u= assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma); 
        [Y, obj_F] = solve_Y(L_u, Y);
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);

        % (2) 更新u
        idx = trace_vec < tau;
        u(idx) = 1 - (1/tau) * trace_vec(idx);

%         % (3) 更新绘图
%         figure(fig);
%         subplot(1,2,1);
%         h_trace.Data = trace_vec;   % 更新 trace 的直方图

%         subplot(1,2,2);
%         h_u.Data = u;               % 更新 u 的直方图
%         ylim([0 1]);                % 固定 y 范围在 [0,1]
%         xlim([0 1]);                % x 轴范围固定 0~1
% 
%         sgtitle(['Iteration ' num2str(it)]);  % 总标题
%         drawnow;

        % (4) 计算目标函数与记录
        fval = sum(trace_vec) - tau * sum(u);
        [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Algorithm 2.3 iter ' num2str(it), ...
              ': Acc=' num2str(result.Accuracy), ...
              ', fval=' num2str(fval)]);
        history.obj(it) = fval;
        history.Accuracy(it) = result.Accuracy;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
        history.tau(it) = tau;

        if tau<tau_max
            tau = tau * taudelt; 
        end
    end

    counts = histcounts(labels_pred, 1:c+1);
    fprintf('最终每个簇的样本数:\n');
    disp(counts);
    plot_history(history, '2.3');
    %plot(history.tau);
      %  rng(s_rng);
end
