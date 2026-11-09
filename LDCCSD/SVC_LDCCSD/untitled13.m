% 测试修正后的 obj_grad_u_3_3 函数
% 创建测试数据
rng(123); % 固定随机种子以便复现

% 参数设置
n = 15;  % 样本数
d = 4;   % 特征维度  
c = 3;   % 聚类数
k = 6;   % 邻居数

% 生成测试数据 X (d×n)
X = randn(d, n);

% 生成 one-hot 聚类标签矩阵 Y (c×n)
labels = randi(c, n, 1);
Y = zeros(c, n);
for i = 1:n
    Y(labels(i), i) = 1;
end

% 生成邻居索引矩阵 nbrIdx (k×n)
nbrIdx = zeros(k, n);
for i = 1:n
    % 为每个样本选择 k 个邻居（包括自己）
    all_indices = 1:n;
    distances = sum((X - X(:,i)).^2, 1);
    [~, sorted_idx] = sort(distances);
    nbrIdx(:, i) = sorted_idx(1:k)';
end

% 中心化矩阵 H (k×k)
H = eye(k) - ones(k,k)/k;

% 其他参数
gamma = 0.05;
tau = 0.8;
u = 0.3 + 0.4 * rand(n, 1); % 权重在 [0.3, 0.7] 之间

fprintf('=== 测试修正后的 obj_grad_u_3_3 函数 ===\n');
fprintf('数据维度: X(%d×%d), Y(%d×%d), nbrIdx(%d×%d)\n', d, n, c, n, k, n);
fprintf('参数: gamma=%.3f, tau=%.3f\n', gamma, tau);

try
    % 调用修正后的函数
    [f, g] = obj_grad_u_3_3(u, X, Y, nbrIdx, H, gamma, tau);
    
    fprintf('\n✓ 函数调用成功！\n');
    fprintf('目标函数值: f = %.6f\n', f);
    fprintf('梯度范数: ||g|| = %.6f\n', norm(g));
    fprintf('梯度维度: %d (期望: %d)\n', length(g), n);
    
    % 检查是否有数值问题
    if any(~isfinite([f; g]))
        warning('发现非有限值！');
    end
    
    % 详细的数值梯度验证
    fprintf('\n=== 数值梯度验证 ===\n');
    epsilon = 1e-6;
    max_test = min(8, n); % 测试前8个分量
    errors = zeros(max_test, 1);
    rel_errors = zeros(max_test, 1);
    
    for i = 1:max_test
        % 前向差分
        u_plus = u; 
        u_plus(i) = u_plus(i) + epsilon;
        f_plus = obj_grad_u_3_3(u_plus, X, Y, nbrIdx, H, gamma, tau);
        
        % 后向差分
        u_minus = u; 
        u_minus(i) = u_minus(i) - epsilon;
        f_minus = obj_grad_u_3_3(u_minus, X, Y, nbrIdx, H, gamma, tau);
        
        % 中心差分
        g_numerical = (f_plus - f_minus) / (2 * epsilon);
        
        % 计算误差
        error_abs = abs(g(i) - g_numerical);
        error_rel = error_abs / (abs(g(i)) + 1e-10);
        
        errors(i) = error_abs;
        rel_errors(i) = error_rel;
        
        % 判断状态
        status = '';
        if error_rel < 1e-5
            status = '✓ 极好';
        elseif error_rel < 1e-3
            status = '✓ 良好';  
        elseif error_rel < 1e-1
            status = '⚠ 可接受';
        else
            status = '✗ 较差';
        end
        
        fprintf('u[%d]: 解析=%.6f, 数值=%.6f, 误差=%.2e, 相对误差=%.2e %s\n', ...
                i, g(i), g_numerical, error_abs, error_rel, status);
    end
    
    % 总体评估
    mean_rel_error = mean(rel_errors);
    max_rel_error = max(rel_errors);
    
    fprintf('\n=== 梯度质量评估 ===\n');
    fprintf('平均相对误差: %.2e\n', mean_rel_error);
    fprintf('最大相对误差: %.2e\n', max_rel_error);
    
    if max_rel_error < 1e-4
        fprintf('🎉 梯度计算质量：优秀\n');
    elseif max_rel_error < 1e-2
        fprintf('✅ 梯度计算质量：良好\n'); 
    elseif max_rel_error < 1e-1
        fprintf('⚠️  梯度计算质量：可接受\n');
    else
        fprintf('❌ 梯度计算质量：需要改进\n');
    end
    
    % 验证目标函数组成部分
    fprintf('\n=== 目标函数分解验证 ===\n');
    trace_part = sum(u .* compute_trace_part(u, X, Y, nbrIdx, H, gamma));
    reg_part = tau * sum(u);
    expected_f = trace_part - reg_part;
    
    fprintf('迹部分: %.6f\n', trace_part);
    fprintf('正则化部分: %.6f\n', reg_part);
    fprintf('期望目标值: %.6f\n', expected_f);
    fprintf('实际目标值: %.6f\n', f);
    fprintf('差异: %.2e\n', abs(f - expected_f));
    
    if abs(f - expected_f) < 1e-12
        fprintf('✓ 目标函数计算完全正确\n');
    end
    
    fprintf('\n🎯 测试完成！\n');
    
catch ME
    fprintf('❌ 测试失败: %s\n', ME.message);
    fprintf('错误位置: %s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

function trace_part = compute_trace_part(u, X, Y, nbrIdx, H, gamma)
    % 辅助函数：独立计算迹部分用于验证
    [d, n] = size(X);
    [k, ~] = size(nbrIdx);
    c = size(Y, 1);
    cols = nbrIdx(:);
    
    % 重新计算所有中间量
    X_tensor = reshape(X(:, cols), d, k, n);
    K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);
    
    H_tensor = repmat(H, 1, 1, n);
    Y_kcn = permute(reshape(Y(:, cols), c, k, n), [2 1 3]);
    class_counts = sum(Y_kcn, 1);
    inv_sqrt_counts = 1 ./ (sqrt(class_counts) + eps);
    Y_kcn_norm = Y_kcn .* repmat(inv_sqrt_counts, [k, 1, 1]);
    B_tensor = pagemtimes(H_tensor, Y_kcn_norm);
    S_tensor = pagemtimes(B_tensor, permute(B_tensor, [2 1 3]));
    
    u_neighbors = reshape(u(nbrIdx), k, n);
    diag_u_neighbors = zeros(k, k, n);
    for i=1:n
        diag_u_neighbors(:,:,i) = diag(u_neighbors(:,i));
    end
    M_tensor = pagemtimes(diag_u_neighbors, pagemtimes(K_tensor, diag_u_neighbors)) + gamma * repmat(eye(k),1,1,n);
    
    Minv_S = pagemldivide(M_tensor, S_tensor);
    trace_part = squeeze(sum(sum(Minv_S .* repmat(eye(k), 1, 1, n), 1), 2));
end
