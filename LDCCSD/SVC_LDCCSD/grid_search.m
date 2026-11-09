function results_table = grid_search(X, Y_true, params, k_list, tau_list, taudelt_list, gamma_list, nRepeat)
% 网格搜索 Algorithm 1.3 (v2) 在不同 k, tau, taudelt, gamma 下的效果
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
                    
                    % ===== Algorithm 1.3 (v2) =====
                    result_aio_13 = [];
                    for iRepeat = 1:nRepeat
                        [labels_pred13, ~] = objective_1_3_v2(X, Y_true, params_local);
                        [~, result_i] = my_eval_y(labels_pred13, Y_true);
                        result_aio_13 = [result_aio_13; result_i']; %#ok<AGROW>
                    end
                    result_avg_13 = mean(result_aio_13, 1);
                    res13 = result_avg_13(1:2);  % 提取 Accuracy 和 NMI
                    
                    % ===== Algorithm 2.3 =====
                    result_aio_23 = [];
                    for iRepeat = 1:nRepeat
                        [labels_pred23, ~] = objective_2_3(X, Y_true, params_local);
                        [~, result_i] = my_eval_y(labels_pred23, Y_true);
                        result_aio_23 = [result_aio_23; result_i']; %#ok<AGROW>
                    end
                    result_avg_23 = mean(result_aio_23, 1);
                    res23 = result_avg_23(1:2);  % 提取 Accuracy 和 NMI


                     % ===== Algorithm 2.3f1 =====
                    result_aio_23f1 = [];
                    for iRepeat = 1:nRepeat
                        [labels_pred23f1, ~] =  objective_2_3_f1(X, Y_true, params_local);
                        [~, result_i] = my_eval_y(labels_pred23f1, Y_true);
                        result_aio_23f1 = [result_aio_23f1; result_i']; %#ok<AGROW>
                    end
                    result_avg_23f1 = mean(result_aio_23f1, 1);
                    res23f1 = result_avg_23f1(1:2);  % 提取 Accuracy 和 NMI


                    % ===== Algorithm 3.3 =====
                    result_aio_33 = [];
                    for iRepeat = 1:nRepeat
                        [labels_pred33, ~] = objective_3_3(X, Y_true, params_local);
                        [~, result_i] = my_eval_y(labels_pred33, Y_true);
                        result_aio_33 = [result_aio_33; result_i']; %#ok<AGROW>
                    end
                    result_avg_33 = mean(result_aio_33, 1);
                    res33 = result_avg_33(1:2);  % 提取 Accuracy 和 NMI
                    


                     % ===== Algorithm 3.3f1 =====
                    result_aio_33f1 = [];
                    for iRepeat = 1:nRepeat
                        [labels_pred33f1, ~] = objective_3_3_f1(X, Y_true, params_local);
                        [~, result_i] = my_eval_y(labels_pred33f1, Y_true);
                        result_aio_33f1 = [result_aio_33f1; result_i']; %#ok<AGROW>
                    end
                    result_avg_33f1 = mean(result_aio_33f1, 1);
                    res33f1 = result_avg_33f1(1:2);  % 提取 Accuracy 和 NMI
                    % ===== 存储结果 =====
                    results = [results; ...
                        k_val, tau_val, taudelt_val, gamma_val, ...
                        res13, res23,res23f1, res33,res33f1]; %#ok<AGROW>
                    fprintf('=== Testing Complete ===\n');
                    
                    % 打印结果
                    fprintf('k=%d, tau=%.3f, taudelt=%.3f, gamma=%.3f | Acc(1.3)=%.4f, Acc(2.3)=%.4f, Acc(2.3f1)=%.4f,Acc(3.3)=%.4f，Acc(3.3f1)=%.4f\n', ...
                        k_val, tau_val, taudelt_val, gamma_val, res13(1), res23(1),res23f1(1), res33(1),res33f1(1));
                end
            end
        end
    end

    % ===== 转换为表格 =====
    colnames = {'k','tau','taudelt','gamma', ...
        'Acc_13','NMI_13', ...
        'Acc_23','NMI_23', ...
        'Acc_23f1','NMI_23f1', ...
        'Acc_33','NMI_33', ...
        'Acc_33f1','NMI_33f1'};
    
    results_table = array2table(results, 'VariableNames', colnames);
    
    % 打印表格到命令行
    disp(results_table); 

    % 保存结果到文件
    save('grid_search_results.mat', 'results_table'); 
    fprintf('results_table 已成功保存为 grid_search_results.mat\n');
end
