function verify_gradient_obj_3_3(u, X, Y, nbrIdx, H, gamma, tau)
    % 计算解析梯度
    [f_analytical, g_analytical] = obj_grad_u_3_3(u, X', Y', nbrIdx', H, gamma, tau);
    
    % 计算数值梯度
    epsilon = 1e-6;
    n = length(u);
    g_numerical = zeros(n, 1);
    
    for i = 1:n
        % 对第i个分量加扰动
        u_plus = u;
        u_plus(i) = u_plus(i) + epsilon;
        [f_plus, ~] = obj_grad_u_3_3(u_plus, X, Y, nbrIdx, H, gamma, tau);
        
        u_minus = u;
        u_minus(i) = u_minus(i) - epsilon;
        [f_minus, ~] = obj_grad_u_3_3(u_minus, X, Y, nbrIdx, H, gamma, tau);
        
        g_numerical(i) = (f_plus - f_minus) / (2 * epsilon);
    end
    
    % 比较解析梯度和数值梯度
    diff = abs(g_analytical - g_numerical);
    max_diff = max(diff);
    relative_error = max_diff / (max(abs(g_analytical)) + 1e-10);
    
    fprintf('梯度验证结果：\n');
    fprintf('最大绝对误差: %.2e\n', max_diff);
    fprintf('最大相对误差: %.2e\n', relative_error);
    
    if max_diff < 1e-5
        fprintf('✓ 梯度计算正确\n');
    elseif max_diff < 1e-3
        fprintf('⚠ 梯度可能有小误差\n');
    else
        fprintf('✗ 梯度计算有明显错误\n');
    end
    
    % 显示前10个分量的对比
    fprintf('\n前10个分量对比：\n');
    fprintf('Index\tAnalytical\tNumerical\tDiff\n');
    for i = 1:min(10, n)
        fprintf('%d\t%.6f\t%.6f\t%.2e\n', i, g_analytical(i), g_numerical(i), diff(i));
    end
    
    % 可视化对比
    figure;
    subplot(2,1,1);
    plot(g_analytical, 'b-', 'LineWidth', 1.5); hold on;
    plot(g_numerical, 'r--', 'LineWidth', 1.5);
    legend('解析梯度', '数值梯度');
    title('梯度对比');
    
    subplot(2,1,2);
    semilogy(diff + 1e-15, 'g-', 'LineWidth', 1.5);
    title('绝对误差 (log scale)');
    ylabel('|解析梯度 - 数值梯度|');
end
