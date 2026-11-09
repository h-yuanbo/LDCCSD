function plot_u_activity(U_hist, easy_idx, hard_idx, tau_hist, thr)
% U_hist : n×T，每列是一轮迭代的 u
% easy_idx / hard_idx : 易/难样本的下标（若 hard_idx 为空，自动取补集）
% tau_hist : 可选，长度 T 的 tau 序列；若不传则不画右轴
% thr : 可选，活跃阈值（默认 0.5）

if nargin < 5 || isempty(thr), thr = 0.5; end
n = size(U_hist,1); T = size(U_hist,2);
if nargin < 3 || isempty(hard_idx)
    all_idx = 1:n; hard_idx = setdiff(all_idx, easy_idx);
end

active_easy = sum(U_hist(easy_idx,:) > thr, 1);
active_hard = sum(U_hist(hard_idx,:) > thr, 1);
t = 1:T;

figure; hold on; grid on; box on
yyaxis left
p1 = plot(t, active_easy, '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
p2 = plot(t, active_hard, '-s', 'LineWidth', 1.6, 'MarkerSize', 5);
ylabel('活跃样本数'); xlabel('迭代/选择次数')
legend([p1 p2], {'易样本活跃数','难样本活跃数'}, 'Location','northwest')

% 可选：右轴画 tau
if nargin >= 4 && ~isempty(tau_hist)
    yyaxis right
    p3 = plot(t, tau_hist(:).', '--', 'LineWidth', 1.5);
    ylabel('Tau 值')
    legend([p1 p2 p3], {'易样本活跃数','难样本活跃数','Tau'}, 'Location','northwest')
end

title('u 的易/难样本活跃度变化')
end
