function [F, obj] = solve_Y_balanced(K, F, maxIter)

if nargin < 3, maxIter = 8; end

n = size(F, 1);
c = size(F, 2);
fKf = sum(F .* (K * F))';
ff = sum(F)';  
m_all = vec2ind(F')';

for j = 1:maxIter
    converged = true;
    order = randperm(n); % 关键：随机化更新顺序
    
    % 限制单轮迁入量
    max_inflow = ceil(n / c * 0.25); 
    inflow_count = zeros(c, 1);

    for idx = 1:n
        i = order(idx);
        m = m_all(i);
        if ff(m) <= 2, continue; end % 保护极小簇
        
        % 计算delta（保持原有逻辑）
        Y_A = F' * K(:, i);
        fKf_s = fKf + 2 * Y_A + K(i, i);
        fKf_s(m) = fKf(m);
        ff_k = ff + 1;
        ff_k(m) = ff(m);
        
        fKf_0 = fKf;
        fKf_0(m) = fKf(m) - 2 * Y_A(m) + K(i, i);
        ff_0 = ff;
        ff_0(m) = ff(m) - 1;
        
        delta = fKf_s ./ ff_k - fKf_0 ./ ff_0;
        [delta_min, p] = min(delta);
        
        % 增加迁移条件限制
        if p ~= m && delta_min < -1e-8 && inflow_count(p) < max_inflow
            converged = false;
            inflow_count(p) = inflow_count(p) + 1;
            
            fKf([m, p]) = [fKf_0(m), fKf_s(p)];
            ff([m, p]) = [ff_0(m), ff_k(p)];
            F(i, [p, m]) = [1, 0];
            m_all(i) = p;
        end
    end
    
    if converged, break; end
end

obj = sum(fKf ./ ff);
end
