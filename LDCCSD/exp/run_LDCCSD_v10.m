clc;clear;
clc;clear;


data_path = fullfile(pwd, '..',  filesep, "data", filesep);
addpath(data_path);
...

data_path = fullfile(pwd, '..',  filesep, "data", filesep);%fullfile生成路径
addpath(data_path);
lib_path = fullfile(pwd, '..',  filesep, "lib", filesep);
addpath(lib_path);
code_path = genpath(fullfile(pwd, '..',  filesep, 'SVC_LDCCSD'));
addpath(code_path);

dirop = dir(fullfile(data_path, '*.mat'));%以结构体形式存储该路径的信息
datasetCandi = {dirop.name};

%datasetCandi = {'JAFFE_213n_676d_10c_uni.mat'};
%datasetCandi = {'k1b_2340n_21839d_6c_tfidf_uni.mat'};
%datasetCandi = {'COIL20_1440n_1024d_20c_uni.mat'};
datasetCandi = {'FISH.mat'};

exp_n = 'LDCCSD';
% profile off;
% profile on;
for i1 = 1 : length(datasetCandi)%遍历
    data_name = datasetCandi{i1}(1:end-4);%移除文件扩展名
    dir_name = [pwd, filesep, exp_n, filesep, data_name];%创造路径
    try
        if ~exist(dir_name, 'dir')%检查dir_name路径是否为一个目录/文件夹
            mkdir(dir_name);
        end
        prefix_mdcs = dir_name;
    catch
        disp(['create dir: ',dir_name, 'failed, check the authorization']);
    end

    clear X y Y;
    load(data_name);
%      X=fea;
%      Y=gnd;
    y=gt;
    if exist('Y', 'var')%检查变量是否存在
        Y_true =Y;
%        X=X';
    end
    if exist('y', 'var')%检查变量是否存在
        Y_true =y;
       X=X';
    end
    if size(X, 1) ~= size(Y_true, 1)
        Y_true = Y_true';
    end
  %  tabulate(Y_true)

    assert(size(X, 1) == size(Y_true, 1));
    disp(length(unique(Y_true)));

    nCluster = length(unique(Y_true));%聚类数
    

   % ===== 选择数据集 =====
%     nSmp = 400;
%     X = X(1:nSmp,:);
%     Y_true = Y_true(1:nSmp,:);

    % ===== 参数 =====
    params = struct;
    params.k          = 10;
    params.gamma      = 10;
    params.taudelt_prime = 1.01;
    params.taudelt = 1.2;
    params.nCluster = length(unique(Y_true));
    params.verbose    = true;
    params.innerU_MaxIters=3;
    params.tau=0.1;
    params.tau_init=0.5;
    params.alpha=0.5;
    params.tau_prime=0.3;
    params.maxIter = 20;       % 最大迭代次数
    params.tau_max=6;
    gamma_list=[1,10,100];
    k_list   = [10, 15,20];          % k 的候选值
    taudelt_list=[1.01,1.05, 1.1];
    tau_list = [0.01, 0.1,0.5];
    H_k = eye(params.k ) - (1/params.k )*ones(params.k );
%      H_n=eye(nSmp) - (1/nSmp)*ones(nSmp);
%      X=H_n*X;
 %并且还要符合自步
    % ===== 训练 =====
    %===== K-Means 对比实验 =====
%     fprintf('========= K-Means ========= :\n');
%     [labels_km, ~] = litekmeans(X, nCluster, 'Replicates', 10, 'MaxIter', 300);
%     %labels_km = kmeans(X, nCluster, 'Replicates', 10, 'MaxIter', 100); % 聚类
%     [~, result_km] = my_eval_y(labels_km, Y_true); % 计算 ACC/NMI 等
%     disp(['K-Means ACC = ', num2str(result_km(1)), ', NMI = ', num2str(result_km(2))]);


    nRepeat = 10;
    
     results_table = grid_search(X, Y_true, params, k_list, tau_list,taudelt_list,gamma_list, nRepeat);
%     disp(results_table);
    fprintf('=== Testing Objective Functions ===\n');
    
%     results_table = grid_search_1_3(X, Y_true, params, k_list, gamma_list, nRepeat);
%     disp(results_table);  


%      fprintf('=========Objective 1.3-v2========= :\n');
result_aio_1_3 = [];
result_aio_2_3 = [];

for iRepeat = 1:nRepeat
    % === 每轮实验使用不同随机种子 ===
    rng(iRepeat, 'twister');
    
    % 保存当前随机数状态
    s = rng;  

    % 运行 1.3
    [labels_pred13, history13] = objective_1_3_v2(X, Y_true, params);
    [~, result_i_13] = my_eval_y(labels_pred13, Y_true);
    result_aio_1_3 = [result_aio_1_3; result_i_13']; %#ok

    % === 恢复同一随机状态，保证起点一致 ===
    rng(s);

    % 运行 2.3（与 1.3 相同的随机初始化）
    [labels_pred23, history23] = objective_2_3(X, Y_true, params);
    [~, result_i_23] = my_eval_y(labels_pred23, Y_true);
    result_aio_2_3 = [result_aio_2_3; result_i_23']; %#ok

    fprintf('Repeat %d done.\n', iRepeat);
end

% === 汇总结果 ===
result_avg_1_3 = mean(result_aio_1_3, 1);
result_avg_2_3 = mean(result_aio_2_3, 1);

%     confusionmat(Y_true, labels_pred23);
%     sum(confusionmat(Y_true, labels_pred23))
% 
%  %results_table = grid_search_2_3_f1(X, Y_true, params, k_list, tau_list, taudelt_list, gamma_list, nRepeat);
% 
%  fprintf('=========Objective 2.3f1========= :\n');
%    result_aio_23f1 = [];
%     for iRepeat = 1:nRepeat
%         [labels_pred23f1, ~] =  objective_2_3_f1(X, Y_true, params);
%         
%         %[labels_pred23f1, ~] =  objective_2_3_with_early_stopping_anchor(X, Y_true, params);
%         [~, result_i] = my_eval_y(labels_pred23f1, Y_true);
%         result_aio_23f1 = [result_aio_23f1; result_i']; %#ok<AGROW>
%     end
%     result_avg_23f1 = mean(result_aio_23f1, 1);
%     res23f1 = result_avg_23f1(1:2);  
%     confusionmat(Y_true, labels_pred23f1);
   

   %     results_table = grid_search_3_3(X, Y_true, params, k_list, tau_list, taudelt_list, gamma_list, nRepeat);


        fprintf('=========Objective 3.3========= :\n');
%     result_aio_3_3 = [];
%     for iRepeat = 1:nRepeat
%         %[labels_pred33, history33] = objective_3_3(X, Y_true, params);
%         [labels_pred33, history33] =objective_3_3_f2(X, Y_true, params);
%          %[labels_pred33, history33] =objective_3_3_protect_safe(X, Y_true, params);
%         [~, result_i] = my_eval_y(labels_pred33, Y_true);
%         result_aio_3_3 = [result_aio_3_3; result_i']; %#ok
%     end
%     result_avg_3_3 = mean(result_aio_3_3, 1);

%  fprintf('=========Objective 3.3f1========= :\n');
%     result_aio_33f1 = [];
%     for iRepeat = 1:nRepeat
%          [labels_pred33f1, ~] = objective_3_3_f1(X, Y_true, params);
%          [~, result_i] = my_eval_y(labels_pred33f1, Y_true);
%          result_aio_33f1 = [result_aio_33f1; result_i']; 
%     end
%     result_avg_33f1 = mean(result_aio_33f1, 1);
    fprintf('=== Testing Complete ===\n');
end

rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);
