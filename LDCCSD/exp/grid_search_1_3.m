function results_table = grid_search_1_3(X, Y_true, params, k_list, gamma_list, nRepeat)
% 网格搜索 1.3 版本，在不同 k 和 gamma 下的效果
%
% 输入:
%   X             - 输入数据 (n x d)
%   Y_true        - 真实标签 (n x 1)
%   params        - 参数结构体 (至少包含 nCluster, gamma, maxIter 等)
%   k_list        - 候选的 k 值数组
%   gamma_list    - 候选的 gamma 数组
%   nRepeat       - 每组参数重复运行次数
%
% 输出:
%   results_table - 保存结果的表格 (k, gamma, Accuracy, NMI)

    results = [];  % 存储结果的数组

    % 遍历所有 k 和 gamma 的组合
    for k_val = k_list
        for gamma_val = gamma_list
            % 拷贝参数并设置当前搜索值
            params_local = params;
            params_local.k     = k_val;
            params_local.gamma = gamma_val;

            % ===== 1.3(v2) =====
            result_aio_13 = [];
            for iRepeat = 1:nRepeat
                % 调用 objective_1_3_v2 函数进行实验
                [labels_pred13, ~] = objective_1_3_v2(X, Y_true, params_local);
                
                % 评估结果
                [~, result_i] = my_eval_y(labels_pred13, Y_true);
                
                % 将每次重复的结果存储
                result_aio_13 = [result_aio_13; result_i']; %#ok<AGROW>
            end
            % 计算平均结果
            result_avg_13 = mean(result_aio_13, 1);
            res13 = result_avg_13(1:5);  % 只取 Accuracy 和 NMI

                      % ===== 存储结果 =====
            results = [results; k_val, gamma_val, res13]; %#ok<AGROW>
            fprintf('=== Testing Complete ===\n');
            
            % *** 建议修改：更新打印结果，显示更多指标 ***
            fprintf('k=%d, gamma=%.3f | Acc(1.3)=%.4f, NMI(1.3)=%.4f, Purity(1.3)=%.4f, AR(1.3)=%.4f, RI(1.3)=%.4f\n', ...
                k_val, gamma_val, res13(1), res13(2), res13(3), res13(4), res13(5));
        end
    end

    % ===== 转换为表格 =====
    % *** 关键修改：更新列名以匹配 5 个指标 ***
    colnames = {'k', 'gamma', 'Acc_13', 'NMI_13', 'Purity_13', 'AR_13', 'RI_13'};
    results_table = array2table(results, 'VariableNames', colnames);
    save('grid_search_1_3_coll20_results.mat', 'results_table'); 
     fprintf('results_table 已成功保存为 grid_search_results.mat\n');
end
