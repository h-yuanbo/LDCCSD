function [F, obj] = solve_Y_selective(K, F, threshold)
    if nargin < 3, threshold = 100; end
    
    fKf = sum(F .* (K * F))';
    ff = sum(F, 1)'; % 修正这里
    
    m_all = vec2ind(F')';
    
    for j = 1:5 % 减少迭代次数
        for i = 1:size(F, 1)
            m = m_all(i);
            if ff(m) == 1
                continue;
            end
            
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
            
            [min_delta, p] = min(delta);
            current_delta = 0; % 保持不动的代价
            
            % 只有当改进足够大时才移动
            if p ~= m && (current_delta - min_delta) > threshold
                fKf([m, p]) = [fKf_0(m), fKf_s(p)];
                ff([m, p]) = [ff_0(m), ff_k(p)];
                F(i, [p, m]) = [1, 0];
                m_all(i) = p;
            end
        end
    end
    
    obj = sum(fKf ./ ff);
end
