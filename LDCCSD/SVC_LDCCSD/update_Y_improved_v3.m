function [F] = update_Y_improved(U, F)
    [n,c] = size(F);
    target_size=n/c;
    % 记录初始簇大小，作为参考
    initial_sizes = sum(F, 1);
     min_cluster_size = max(5, floor(n/(c*0.7))); % 35而不是50 % 最小簇大小阈值
    penalty_threshold=target_size*1.3;
    for iterf = 1:5
        converged = true;
        
        % 计算当前状态的增益
        ff = sum(F, 1);           % 各簇大小
        uf = sum(U .* F, 1);      % 各簇的加权和
        
        for i = 1:n
            ui = U(i,:);
            [~, id0] = find(F(i,:) == 1);
            
            % 保护小簇：如果移出后簇会变得过小，跳过
            if ff(id0) <= min_cluster_size
                continue;
            end
            
            % 计算移动到各簇的增益
            best_gain = -Inf;
            best_cluster = id0;
            
            for k = 1:c
                if k == id0
                    continue;
                end
                
                % 计算增益时考虑簇大小限制
                gain_k = (uf(k) + ui(k))/sqrt(ff(k) + 1) - uf(k)/sqrt(ff(k) + eps);
                loss_id0 = uf(id0)/sqrt(ff(id0)) - (uf(id0) - ui(id0))/sqrt(ff(id0) - 1);
                
                total_gain = gain_k - loss_id0;
                
                % 添加簇大小惩罚，防止过度不平衡
                size_penalty = 0;
                if ff(k) > penalty_threshold  % 如果目标簇已经很大
                    size_penalty = 0.2 * (ff(k) - target_size) / target_size;
                end
                
                adjusted_gain = total_gain - size_penalty;
                
                if adjusted_gain > best_gain
                    best_gain = adjusted_gain;
                    best_cluster = k;
                end
            end
            
            % 只有显著改进才移动
            if best_cluster ~= id0 && best_gain > 1e-6
                converged = false;
                F(i, id0) = 0;
                F(i, best_cluster) = 1;
                
                % 更新统计量
                ff(id0) = ff(id0) - 1;
                ff(best_cluster) = ff(best_cluster) + 1;
                uf(id0) = uf(id0) - ui(id0);
                uf(best_cluster) = uf(best_cluster) + ui(best_cluster);
            end
        end
        
    cluster_sizes = sum(F, 1);
target_size = n / c;


end