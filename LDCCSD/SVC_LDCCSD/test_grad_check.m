%% 参数设置
d = 5;    % 特征维度
n = 20;   % 样本数
c = 3;    % 标签/约束维度
k = 4;    % 邻居数

gamma = 0.1;
tau   = 0.2;

%% 随机生成数据
rng(0);  % 固定随机种子，保证可复现
X = randn(d, n);       % d×n
Y = randn(c, n);       % c×n
G = randn(c, n);       % c×n
H = randn(k, k);       % k×k
nbrIdx = randi(n, k, n);  % 邻居索引，范围在 1..n

%% 初始化 u
u = rand(n, 1);

%% 调用你的函数，得到 f 和梯度 g
[f, g] = obj_grad_u2(u, X, Y, G, nbrIdx, H, gamma, tau);

fprintf('目标函数值 f = %.6f\n', f);
fprintf('梯度范数 ||g|| = %.6f\n', norm(g));

%% 数值梯度检查
grad_check = zeros(size(u));
eps0 = 1e-6;

% 这里只检查前 10 个分量
for i = 1:min(10, n)
    e = zeros(size(u)); 
    e(i) = 1;
    
    f1 = obj_grad_u2(u + eps0 * e, X, Y, G, nbrIdx, H, gamma, tau);
    f2 = obj_grad_u2(u - eps0 * e, X, Y, G, nbrIdx, H, gamma, tau);
    
    grad_check(i) = (f1 - f2) / (2 * eps0);
end

%% 打印对比
disp('   解析梯度 g(1:10)      数值梯度 grad_check(1:10)')
disp([g(1:10) grad_check(1:10)]);
