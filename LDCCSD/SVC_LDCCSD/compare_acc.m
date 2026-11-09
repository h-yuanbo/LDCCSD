function compare_acc(filepath)
    % 载入 Excel 表格
    T = readtable(filepath);

    % 实验编号
    exp_ids = {'13','23','33'};
    
    % 创建图形
    figure;
    hold on;
    
    % 设置线型和颜色
    colors = ['r', 'g', 'b']; % 红色，绿色，蓝色
    linestyles = {'-o', '-s', '-^'}; % 线条样式

    % 遍历三个实验，画出 Acc 对比
    for i = 1:length(exp_ids)
        id = exp_ids{i};
        
        % 获取 Acc 列名
        acc_col = ['Acc_' id];
        
        % 提取数据
        acc_data = T.(acc_col);
        
        % 绘制 Acc 曲线
        plot(acc_data, linestyles{i}, 'LineWidth', 1.5, 'Color', colors(i));
    end
    
    % 设置图形属性
    xlabel('实验索引');
    ylabel('Acc 值');
    title('1.3, 2.3, 3.3 的 Acc 对比');
    legend('Acc 1.3', 'Acc 2.3', 'Acc 3.3', 'Location', 'best');
    grid on;
    hold off;
end
