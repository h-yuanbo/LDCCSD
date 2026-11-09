function newL2 = bestMap(L1, L2)
% bestMap: permute labels of L2 to match L1 as good as possible
% L1: ground truth labels
% L2: clustering labels

L1 = L1(:);
L2 = L2(:);

if size(L1) ~= size(L2)
    error('size(L1) must == size(L2)');
end

Label1 = unique(L1);
nClass1 = length(Label1);

Label2 = unique(L2);
nClass2 = length(Label2);

nClass = max(nClass1, nClass2);

% 生成计数矩阵
G = zeros(nClass);
for i = 1:nClass1
    for j = 1:nClass2
        G(i,j) = sum(L1 == Label1(i) & L2 == Label2(j));
    end
end

% 调用 munkres 匈牙利算法求解最佳匹配
[~, matching] = munkres(-G);

% 初始化输出标签
newL2 = zeros(size(L2));

for j = 1:nClass2
    match_idx = matching(j);
    if match_idx > 0   % 只有大于 0 的索引才有效
        newL2(L2 == Label2(j)) = Label1(match_idx);
    else
        % 没有匹配的列，可选择赋一个默认类别，比如保持原标签
        newL2(L2 == Label2(j)) = Label1(1);
    end
end

end
