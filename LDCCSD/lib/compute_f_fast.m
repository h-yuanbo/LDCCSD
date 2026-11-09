function [f, p, r] = compute_f_fast(T, H)
    if length(T) ~= length(H)
        error('T and H must have the same length.');
    end

    % 计算 numT（真实标签对的数量）
    [~, ~, uT] = unique(T(:));
    countsT = accumarray(uT, 1);
    numT = sum(countsT .* (countsT - 1) / 2);

    % 计算 numH（预测标签对的数量）
    [~, ~, uH] = unique(H(:));
    countsH = accumarray(uH, 1);
    numH = sum(countsH .* (countsH - 1) / 2);

    % 计算 numI（共同标签对的数量）
    [~, ~, uTH] = unique([T(:), H(:)], 'rows');
    countsTH = accumarray(uTH, 1);
    numI = sum(countsTH .* (countsTH - 1) / 2);

    % 计算精确率 (p) 和召回率 (r)
    p = 1;
    if numH > 0
        p = numI / numH;
    end

    r = 1;
    if numT > 0
        r = numI / numT;
    end

    % 计算 F1 分数
    if (p + r) == 0
        f = 0;
    else
        f = 2 * (p * r) / (p + r);
    end
end