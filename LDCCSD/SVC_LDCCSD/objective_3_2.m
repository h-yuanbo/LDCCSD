function [labels_pred,Y,history] = objective_3_2(X,Y_true , params)
% ------------------------------------------------------------
% Correlated Self-Paced LDC with Label-Guided Alignment
% 算法本体：交替更新 G, Y, u，并在结尾“解冻”u
% 输入:
%   X (n×d)       : 数据
%   Y_true (n×1)  : 仅用于确定簇数 c（不在此处做评估）
%   params        : 结构体(见 set_default_params)
% 输出:
%   labels_pred   : 预测标签（由最终 Y 得到）
%   Y, G, u       : 最终解
%   history       : 训练过程信息（目标值轨迹、tau轨迹、u统计）
%   aux           : 其它有用缓存（邻接、H、最后一次 L_u、tau 等）
% ------------------------------------------------------------

% ---- 读取/补齐参数 ----


% ---- 基本量 ----
n  = size(X,1);
c=params.nCluster;
k  = params.k;
H  = eye(k) - (1/k)*ones(k);
tau=params.tau;
nbrIdx = build_knn_graph(X, k);
% ---- 初始化   ----
u = 0.1 + 0.15*rand(n, 1);  % 范围[0.05, 0.2]
L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);

lab = randi(c, [n,1]);
Y = full(sparse((1:n)', lab, 1, n, c));
result= my_eval_y(lab, Y_true);
disp(['init Label ', num2str(result.Accuracy), ]);


% disp('cluster sizes in Y:'), disp(sum(Y,1))
%L_u = assemble_Lu_v2(X, H, nbrIdx, u, params.gamma);


G = Y * diag(sum(Y,1).^(-.5));
fval=0;
% ---- 外层迭代 ----
for it = 1:params.maxIter

    %L_u = assemble_Lu(X, H, nbrIdx, u, params.gamma);
    % [G]   = update_G(L_u, Y);
    % (2) Y
%     innerIterY = it;

    innerIterY = 1;
    %if it>10

    Q = L_u * G;
   
    %U = -Q;
    %Y = update_Y_improved(U, Y, innerIterY);
    %[Y, obj_F] = update_Y_min(Q, Y, innerIterY);
    [Y, obj_F] = solve_Y(L_u, Y);
    [~, labels_pred] = max(Y, [], 2);
    result= my_eval_y(labels_pred, Y_true);
    disp(['Algorithm 3.2 iter', num2str(it), ': ', num2str(result.Accuracy), '    ', num2str(obj_F(end))]);


    G = Y * diag(sum(Y,1).^(-.5));

    % (3) u  +  tau（预算约束）
    options = optimoptions('fmincon', ...
        'SpecifyObjectiveGradient', true, ...
        'Algorithm','sqp', ...
        'HessianApproximation','lbfgs', ...
        'Display','none', ...
        'MaxIterations', params.innerU_MaxIters, ...
        'StepTolerance',1, ...
        'OptimalityTolerance',1e-2);


    
           [u, tau, fval] = update_u_fixed(u, X',Y', G', nbrIdx', H, params.gamma, ...
                                       tau,options);
    
    %fval= objective_value_2(u, X', Y', G', nbrIdx', H, params.gamma, 0);
    % -- 监控/记录 --
    history.obj(it)      = fval;
    history.tau(it)      = tau;
    history.u_stat(it,:) = [min(u) median(u) max(u)];
    tau=tau*params.taudelt;
    
    % 监控

%     if params.verbose
%         fprintf('Iter %2d | tau=%.3f | mean(u)=%.3f | min cluster size=%d\n', ...
%             it, tau, mean(u), min(sum(Y,1)));
%     end
end
%disp(result);


end


