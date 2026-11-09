% 正确性测试
% 测试用例 1: 完全匹配
T1 = [1, 1, 2, 2];
H1 = [1, 1, 2, 2];
[f_old1, p_old1, r_old1] = compute_f(T1, H1);
[f_new1, p_new1, r_new1] = compute_f_fast(T1, H1);
assert(abs(f_old1 - f_new1) < 1e-6 && abs(p_old1 - p_new1) < 1e-6 && abs(r_old1 - r_new1) < 1e-6, 'Test case 1 failed');

% 测试用例 2: 部分匹配
T2 = [1, 2, 1, 2];
H2 = [1, 1, 2, 2];
[f_old2, p_old2, r_old2] = compute_f(T2, H2);
[f_new2, p_new2, r_new2] = compute_f_fast(T2, H2);
assert(abs(f_old2 - f_new2) < 1e-6 && abs(p_old2 - p_new2) < 1e-6 && abs(r_old2 - r_new2) < 1e-6, 'Test case 2 failed');

% 测试用例 3: 无匹配
T3 = [1, 2, 3, 4];
H3 = [1, 2, 3, 4];
[f_old3, p_old3, r_old3] = compute_f(T3, H3);
[f_new3, p_new3, r_new3] = compute_f_fast(T3, H3);
assert(abs(f_old3 - f_new3) < 1e-6 && abs(p_old3 - p_new3) < 1e-6 && abs(r_old3 - r_new3) < 1e-6, 'Test case 3 failed');

disp('所有正确性测试通过。');

% 性能测试
N = 10000; % 测试数据规模
T_large = randi(100, 1, N); % 生成随机标签
H_large = randi(100, 1, N);

% 预热（避免首次运行时的初始化开销）
compute_f(T_large(1:10), H_large(1:10));
compute_f_fast(T_large(1:10), H_large(1:10));

% 测试原始函数时间
tic;
[~, ~, ~] = compute_f(T_large, H_large);
time_old = toc;

% 测试优化函数时间
tic;
[~, ~, ~] = compute_f_fast(T_large, H_large);
time_new = toc;

fprintf('原始函数耗时: %.4f 秒\n', time_old);
fprintf('优化函数耗时: %.4f 秒\n', time_new);
fprintf('加速倍数: %.2f\n', time_old / time_new);