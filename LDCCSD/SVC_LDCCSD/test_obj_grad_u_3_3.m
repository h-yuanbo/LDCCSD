% 创建简单的测试数据
rng(42); % 固定随机种子

% 参数设置
n = 20;  % 样本数
d = 3;   % 特征维度  
c = 3;   % 聚类数
k = 5;   % 邻居数

% 生成测试数据
X = randn(d, n);  % d×n

% 生成one-hot聚类标签矩阵 Y (c×n)
labels = randi(c, n, 1);
Y = zeros(c, n);
for i = 1:n
    Y(labels(i), i) = 1;
end

% 生成邻居索引矩阵 (k×n)
nbrIdx = zeros(k, n);
for i = 1:n
    % 简单地选择前k个不同的邻居（包括自己）
    neighbors = mod((i-1:i+k-2), n) + 1;
    nbrIdx(:, i) = neighbors';
end

% 中心化矩阵 H (k×k)
H = eye(k) - ones(k,k)/k;

% 其他参数
gamma = 0.1;
tau = 1.0;
u = rand(n, 1); % 随机权重

% 调用函数
fprintf('Testing obj_grad_u_3_3...\n');
try
    [f, g] = obj_grad_u_3_3(u, X, Y, nbrIdx, H, gamma, tau);
    
    fprintf('Function call successful!\n');
    fprintf('Objective value: %.6f\n', f);
    fprintf('Gradient norm: %.6f\n', norm(g));
    fprintf('Gradient size: %d\n', length(g));
    
    % 检查梯度维度
    if length(g) ~= n
        error('Gradient dimension mismatch: expected %d, got %d', n, length(g));
    end
    
    % 数值梯度检验
    fprintf('\nPerforming numerical gradient check...\n');
    epsilon = 1e-6;
    
    for i = 1:min(5, n) % 只检查前5个分量以节省时间
        u_plus = u; u_plus(i) = u_plus(i) + epsilon;
        u_minus = u; u_minus(i) = u_minus(i) - epsilon;
        
        f_plus = obj_grad_u_3_3(u_plus, X, Y, nbrIdx, H, gamma, tau);
        f_minus = obj_grad_u_3_3(u_minus, X, Y, nbrIdx, H, gamma, tau);
        
        g_numerical = (f_plus - f_minus) / (2 * epsilon);
        
        error_i = abs(g(i) - g_numerical);
        rel_error_i = error_i / (abs(g(i)) + 1e-8);
        
        fprintf('u[%d]: analytical=%.6f, numerical=%.6f, error=%.2e, rel_error=%.2e\n', ...
                i, g(i), g_numerical, error_i, rel_error_i);
    end
    
    % 检查是否有数值问题
    if any(~isfinite(f)) || any(~isfinite(g))
        warning('Found non-finite values in objective or gradient!');
    end
    
    % 检查目标函数的组成部分
    trace_part = sum(u .* compute_trace_only(u, X, Y, nbrIdx, H, gamma));
    regularization_part = tau * sum(u);
    expected_f = trace_part - regularization_part;
    
    fprintf('\nObjective breakdown:\n');
    fprintf('Trace part: %.6f\n', trace_part);
    fprintf('Regularization part: %.6f\n', regularization_part);
    fprintf('Expected f: %.6f\n', expected_f);
    fprintf('Actual f: %.6f\n', f);
    fprintf('Difference: %.2e\n', abs(f - expected_f));
    
    fprintf('\n✓ Test completed successfully!\n');
    
catch ME
    fprintf('Error occurred: %s\n', ME.message);
    rethrow(ME);
end

function trace_vec = compute_trace_only(u, X, Y, nbrIdx, H, gamma)
    [d, n] = size(X);           
    [k, ~] = size(nbrIdx);     
    c = size(Y, 1);
    cols = nbrIdx(:);
    
    % 使用与obj_grad_u_3_3完全相同的计算步骤
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
    for i = 1:n
        diag_u_neighbors(:,:,i) = diag(u_neighbors(:,i));
    end
    M_tensor = pagemtimes(diag_u_neighbors, pagemtimes(K_tensor, diag_u_neighbors)) + gamma * repmat(eye(k),1,1,n);
    
    Minv_S = pagemldivide(M_tensor, S_tensor);
    trace_vec = squeeze(sum(sum(Minv_S .* repmat(eye(k), 1, 1, n), 1), 2));
end
