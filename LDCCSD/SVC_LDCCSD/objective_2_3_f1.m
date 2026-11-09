function [labels_pred, history] = objective_2_3_f1(X, Y_true, params)
    % Algorithm 2.3 with Self-Paced Learning (logistic weighting)
    % 对应目标函数2.3: min_{u,Y} sum_i u_i * loss_i - beta sum_i u_i
    
    n = size(X, 1);                    % 样本数
    c = params.nCluster;               % 聚类类别数
    k = params.k;                       % k近邻参数
    H = eye(k) - (1/k)*ones(k);        % 中心化矩阵
    tau = params.tau;            % 初始 SPL 门槛 (类似 tau)
    taudelt = params.taudelt;       % 每轮递增系数
    tau_max  = params.tau_max;        % 最大 beta                       % 可调参数

    nbrIdx = build_knn_graph(X, k);

    % ==== 初始化 ====
    u = ones(n, 1);  % 初始全部样本权重为 1
    
    L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
    noise_level = 0.2;
    L_u = L_u + noise_level * randn(size(L_u));
    L_u = (L_u + L_u') / 2; 
    % 谱初始化聚类
    [V, eigenvals] = eig(full(L_u));
    [~, idx] = sort(diag(eigenvals), 'ascend');
    G = real(V(:, idx(2:c+1)));
  [labels_pred, ~] = litekmeans(NormalizeFea(G), c, 'MaxIter', 50, 'Replicates',1);
    idx2 = randi(n, ceil(0.3*n));
     labels_pred(idx2) = randi(c, ceil(0.3*n));
    Y = full(ind2vec(labels_pred')');
    result_init = my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 2.3f1 init: Acc=', num2str(result_init.Accuracy)]);
    tol = 1e-12;  
     trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
    fprintf('[Init Rough] loss: min=%.6f, max=%.6f, std=%.6f, zeros=%d/%d\n', ...
        min(trace_vec), max(trace_vec), std(trace_vec), sum(abs(trace_vec) < tol), numel(trace_vec));
    % 历史记录
    history.obj = zeros(params.maxIter, 1);
    history.tau = zeros(params.maxIter, 1);
    history.u_stat = zeros(params.maxIter, 3);
    history.Accuracy=zeros(params.maxIter, 1);
    % ==== 主循环 ====
    for it = 1:params.maxIter
        % (1) 更新 Y
        L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, params.gamma);
        %trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        %[Y, obj_F] =solve_Y_add(L_u, Y);
        [Y, obj_F] = solve_Y(L_u, Y);
       
        
        % (2) 计算损失 (trace_vec)
        trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, params.gamma);
        trace_all=0;
         for i=1:n
            trace_all=trace_all+trace_vec(i);
         end
         disp(['trace_all=',num2str(trace_all)]);
        fprintf('[Rough] loss: min=%.6f, max=%.6f, std=%.6f, zeros=%d/%d\n', ...
        min(trace_vec), max(trace_vec), std(trace_vec), sum(abs(trace_vec) < tol), numel(trace_vec));
%         if it == 1 || mod(it,5) == 0
%             figure; 
%             histogram(trace_vec, 50);
%             title('Initial Loss Distribution');
%             xlabel('Loss value'); 
%             ylabel('Count');
%         end

        % (3) 和 ADMM 模型相同的 SPL (logistic 权重更新)
        for i = 1:n
            % 逻辑斯蒂函数  [0,1] 权重
            u(i) = 1 / (1 + exp( 2 * ( trace_vec(i) - tau ) ));
        end
        
        % (4) 更新目标值
        fval = sum(u .* trace_vec) - tau * sum(u);
         [~, labels_pred] = max(Y, [], 2);
        result = my_eval_y(labels_pred, Y_true);
        disp(['Iter ', num2str(it), ...
              ': ACC=', num2str(result.Accuracy), ...
              '   ObjF=', num2str(fval)]);
        % (5) 保存记录
        history.obj(it) = fval;
        history.tau(it) = tau;
        history.u_stat(it,:) = [min(u) mean(u) max(u)];
        history.Accuracy(it)=result.Accuracy;
%         if it==1
%             pure_count = 0;
%             for i = 1:n
%                 labs = unique(labels_pred(nbrIdx(i,:)));
%                 if numel(labs) == 1
%                 pure_count = pure_count + 1;
%                 end
%             end
%             fprintf('第一轮邻域纯化比例: %.2f%%\n', 100 * pure_count / n);
%         end
%         if it > 5 && abs(history.Accuracy(it) - history.Accuracy(it-3)) < 0.001
%             % 随机激活一部分冻结样本
%             frozen_idx = find(u == tol);
%             reactivate_idx = randsample(frozen_idx, min(50, length(frozen_idx)));
%             u(reactivate_idx) = 0.1;
%         end
        % (6) 逐步提高 beta (从易到难)

            tau = min(tau * taudelt, tau_max);
    end
        counts = histcounts(labels_pred, 1:c+1);
    fprintf('最终每个簇的样本数:\n');
    disp(counts);
    plot_history(history, '2.3f1');
end
