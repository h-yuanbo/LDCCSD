function results_table = grid_search_3_3(X, Y_true, params, k_list, tau_list, taudelt_list, gamma_list, nRepeat)
% 网格搜索 Algorithm 2.3 在不同 k, tau, taudelt, gamma 下的效果
%
% 输入:
%   X             - 输入数据 (n x d)
%   Y_true        - 真实标签 (n x 1)
%   params        - 参数结构体 (至少包含 nCluster, gamma, maxIter, innerU_MaxIters, tau_beta 等)
%   k_list        - 候选的 k 值数组
%   tau_list      - 候选的 tau 初始值数组
%   taudelt_list  - 候选的 tau 增长因子数组
%   gamma_list    - 候选的 gamma 数组
%   nRepeat       - 每组参数重复运行次数
%
% 输出:
%   results_table - 保存结果的表格 (k, tau, taudelt, gamma, Accuracy, NMI, Purity, AR, RI)

    results = [];  % 存储结果的数组

    % 遍历所有 k, tau, taudelt, gamma 的组合
    for k_val = k_list
        for tau_val = tau_list
            for taudelt_val = taudelt_list
                for gamma_val = gamma_list
                    
                    % 拷贝参数并设置当前搜索值
                    params_local = params;
                    params_local.k       = k_val;
                    params_local.tau     = tau_val;
                    params_local.taudelt = taudelt_val;
                    params_local.gamma   = gamma_val;
                    
                    % ===== 仅针对 Algorithm 2.3 进行实验 =====
                    result_aio_33 = [];
                    for iRepeat = 1:nRepeat
                        % 调用 objective_2_3 函数进行实验
                        [labels_pred33, ~] = objective_3_3(X, Y_true, params_local);
                        
                        % 评估结果 (my_eval_y 返回 res_struct 和 res_column)
                        % 我们需要 res_column 中的各项指标
                        [~, result_i] = my_eval_y(labels_pred33, Y_true);
                        
                        % 将每次重复的结果存储
                        result_aio_33 = [result_aio_33; result_i']; %#ok<AGROW>
                    end
                    % 计算平均结果
                    result_avg_33 = mean(result_aio_33, 1);
                    
                    % *** 关键修改：提取前 5 个指标 (Accuracy, NMI, Purity, AR, RI) ***
                    res33 = result_avg_33(1:5);

                    % ===== 存储结果 =====
                    results = [results; ...
                        k_val, tau_val, taudelt_val, gamma_val, ...
                        res33]; %#ok<AGROW>
                    fprintf('=== Testing Complete ===\n');
                    
                    % 打印结果 (显示所有 5 个指标)
                    fprintf('k=%d, tau=%.3f, taudelt=%.3f, gamma=%.3f | Acc(2.3)=%.4f, NMI(2.3)=%.4f, Purity(2.3)=%.4f, AR(2.3)=%.4f, RI(2.3)=%.4f\n', ...
                        k_val, tau_val, taudelt_val, gamma_val, ...
                        res33(1), res33(2), res33(3), res33(4), res33(5));
                end
            end
        end
    end

    % ===== 转换为表格 =====
    % *** 关键修改：更新列名以匹配 5 个指标 ***
    colnames = {'k','tau','taudelt','gamma', ...
        'Acc_33','NMI_33','Purity_33','AR_33','RI_33'};
    
    results_table = array2table(results, 'VariableNames', colnames);
    
    disp(results_table); % 打印表格到命令行

    save('grid_search_3_3_results.mat', 'results_table'); 
    fprintf('results_table 已成功保存为 grid_search_3_3_results.mat\n');

end
