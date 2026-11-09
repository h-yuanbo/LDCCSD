% 脚本：filter_and_visualize_selected.m

% 1. 设定文件路径
file_path = 'D:\slefpacedclusting\汇报\aff_grid.xlsx';

% 2. 载入 Excel 数据
T = readtable(file_path);

% 3. 筛选符合条件的实验组
filtered_data = T(T.Acc_33 > T.Acc_23 & T.Acc_23 > T.Acc_13 & ...
                  T.NMI_33 > T.NMI_23 & T.NMI_23 > T.NMI_13, :);

% 4. 显示筛选后的数据
disp('符合条件的实验数据：');
disp(filtered_data);

% 5. 如果有符合条件的数据，绘制它们的 Acc 和 NMI 对比图
if ~isempty(filtered_data)
    % 绘制 Acc 对比图
    figure;
    hold on;
    plot(filtered_data.Acc_13, '-o', 'LineWidth', 1.5, 'Color', 'r');
    plot(filtered_data.Acc_23, '-s', 'LineWidth', 1.5, 'Color', 'g');
    plot(filtered_data.Acc_33, '-^', 'LineWidth', 1.5, 'Color', 'b');
    hold off;
    
    % 设置图形属性
    xlabel('实验索引');
    ylabel('Acc 值');
    title('符合条件的实验组 Acc 对比');
    legend('Acc 13', 'Acc 23', 'Acc 33', 'Location', 'best');
    grid on;

    % 绘制 NMI 对比图
    figure;
    hold on;
    plot(filtered_data.NMI_13, '-o', 'LineWidth', 1.5, 'Color', 'r');
    plot(filtered_data.NMI_23, '-s', 'LineWidth', 1.5, 'Color', 'g');
    plot(filtered_data.NMI_33, '-^', 'LineWidth', 1.5, 'Color', 'b');
    hold off;
    
    % 设置图形属性
    xlabel('实验索引');
    ylabel('NMI 值');
    title('符合条件的实验组 NMI 对比');
    legend('NMI 13', 'NMI 23', 'NMI 33', 'Location', 'best');
    grid on;

else
    disp('没有符合条件的实验数据。');
end