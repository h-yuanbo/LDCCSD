function verify_obj_grad_u2()
    % 创建测试数据
    n = 50; d = 10; c = 3; k = 5;
    X = randn(d, n);
    Y = rand(n, c); Y = Y ./ sum(Y, 2);  % 归一化
    G = randn(c, n);
    nbrIdx = randi(n, k, n);
    H = eye(k) - ones(k)/k;
    gamma = 0.1; tau = 0.1;
    u = 0.1 + 0.4*rand(n, 1);  % 随机初始化 u
    
    % 1. 解析梯度
    [f_analytical, g_analytical] = obj_grad_u2(u, X, Y', G, nbrIdx, H, gamma, tau);
    
    % 2. 数值梯度验证
    epsilon = 1e-6;
    g_numerical = zeros(n, 1);
    for i = 1:n
        u_plus = u; u_plus(i) = u_plus(i) + epsilon;
        u_minus = u; u_minus(i) = u_minus(i) - epsilon;
        
        f_plus = obj_grad_u2(u_plus, X, Y', G, nbrIdx, H, gamma, tau);
        f_minus = obj_grad_u2(u_minus, X, Y', G, nbrIdx, H, gamma, tau);
        
        g_numerical(i) = (f_plus - f_minus) / (2 * epsilon);
    end
    
    % 3. 梯度误差比较
    diff = abs(g_analytical - g_numerical);
    max_diff = max(diff);
    rel_diff = max_diff / (max(abs(g_analytical)) + eps);
    
    fprintf('梯度验证结果:\n');
    fprintf('最大绝对误差: %.2e\n', max_diff);
    fprintf('最大相对误差: %.2e\n', rel_diff);
    
    if rel_diff < 1e-5
        fprintf('✓ 梯度正确\n');
    else
        fprintf('✗ 梯度有误\n');
        fprintf('前10个分量比较:\n');
        for i = 1:min(10,n)
            fprintf('i=%d: 解析=%.6f, 数值=%.6f, 误差=%.2e\n', ...
                i, g_analytical(i), g_numerical(i), diff(i));
        end
    end
    
    % 4. 检查目标函数是否沿负梯度方向下降
    alpha = 0.001;
    u_new = u - alpha * g_analytical;
    f_new = obj_grad_u2(u_new, X, Y', G, nbrIdx, H, gamma, tau);

    if f_new < f_analytical
        fprintf('✓ 目标函数沿负梯度方向下降\n');
    else
        fprintf('✗ 目标函数未下降，梯度可能有误\n');
    end
    
    % 5. 特殊情况测试：tau=0时，梯度是否等于trace_vec
    [f1, g1,trace_vec] = obj_grad_u2(u, X, Y', G, nbrIdx, H, gamma, 0);
    if norm(g1 - trace_vec) < 1e-5
        fprintf('✓ 特殊情况测试通过 (tau=0)\n');
    else
        fprintf('✗ 特殊情况测试失败 (tau=0)\n');
    end
    
    % 6. 测试u全为常数时的行为
    u_const = 0.5 * ones(n, 1);
    [f2, g2] = obj_grad_u2(u_const, X, Y', G, nbrIdx, H, gamma, tau);
    fprintf('常数u情况下：f2=%.6f, g2=\n', f2);
    disp(g2);
end
