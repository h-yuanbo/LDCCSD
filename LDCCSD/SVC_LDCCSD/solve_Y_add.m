function [F, obj] = solve_Y_add(K, F)

fKf = sum(F .* (K * F))';
ff = sum(F)';  % 簇大小

m_all = vec2ind(F')';  % 每个样本当前簇索引
iter = 10;  % 可以根据需求运行更多迭代，直到收敛

for j = 1:iter
    for i = 1:size(F, 1)
        m = m_all(i);
        if ff(m) == 1
            % 避免生成空簇
            continue;
        end
    
        % 当前样本与各簇的关联
        Y_A = F' * K(:, i);
    
        % 把样本i加入各簇情况下的 fKf, ff
        fKf_s = fKf + 2 * Y_A + K(i, i);
        fKf_s(m) = fKf(m);
        ff_k = ff + 1;
        ff_k(m) = ff(m);
    
        % 把样本i移出当前簇的 fKf, ff
        fKf_0 = fKf;
        fKf_0(m) = fKf(m) - 2 * Y_A(m) + K(i, i);
        ff_0 = ff;
        ff_0(m) = ff(m) - 1;
    
        % delta = 加入目标簇后的平均值变化
        delta = fKf_s ./ ff_k - fKf_0 ./ ff_0;
    
        % 选择 "目标值下降" 幅度最大的簇
        [delta_min, p] = min(delta);
        if p ~= m && delta_min < 0   % 只有当目标值下降才迁移
            % 更新 fKf 和 ff
            fKf([m, p]) = [fKf_0(m), fKf_s(p)];
            ff([m, p]) = [ff_0(m), ff_k(p)];

            % 更新标签
            F(i, [p, m]) = [1, 0];
            m_all(i) = p;
        end
    end
    obj = sum(fKf ./ ff);
end

end
