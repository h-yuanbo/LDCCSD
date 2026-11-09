function F = update_Y_improved_v2(U, F)
% 目的：最大化 sum_r (q_r^T y_r)/sqrt(|C_r|)，调用时传 -Q
[n,c] = size(F);
ff = sum(F,1);           % 各簇规模 (1×c)
uf = sum(U.*F,1);        % 各簇加权和 (1×c)

% —— 保险参数（可微调）——
target   = n/c;
min_sz   = max(3, floor(0.6*target));   % 最小簇规模
max_sz   = ceil(1.8*target);            % 最大簇规模
cap_in   = max(1, round(0.10*target));  % 每簇每轮最多迁入数量
gain_thr = 1e-6;                        % 只接受显著正增益

maxIter = 3;                             % 每次调用小步更新
for it = 1:maxIter
    moved_in = zeros(1,c);               % 本轮各簇迁入计数
    changed = 0;

    for i = 1:n
        % 当前所属簇
        id0 = find(F(i,:)==1,1);
        if ff(id0) <= min_sz, continue; end  % 保护小簇，不许再挖

        ui = U(i,:);

        % 先算移出原簇的损失（与目的簇无关，O(1)）
        loss0 = uf(id0)/sqrt(ff(id0)+eps) - (uf(id0)-ui(id0))/sqrt(ff(id0)-1+eps);

        % 逐簇候选：目的簇不得超上限，且本轮未超 cap
        best_gain = -inf; best_k = id0;
        for k = 1:c
            if k==id0, continue; end
            if ff(k) >= max_sz,    continue; end
            if moved_in(k) >= cap_in, continue; end

            gain_k = (uf(k)+ui(k))/sqrt(ff(k)+1+eps) - uf(k)/sqrt(ff(k)+eps);
            total_gain = gain_k - loss0;      % 真正的“加入-移出”净增益

            if total_gain > best_gain
                best_gain = total_gain; best_k = k;
            end
        end

        % 只有净增益足够大才迁移
        if best_k ~= id0 && best_gain > gain_thr
            % 执行迁移
            F(i,id0)=0; F(i,best_k)=1;
            ff(id0)=ff(id0)-1;  ff(best_k)=ff(best_k)+1;
            uf(id0)=uf(id0)-ui(id0); uf(best_k)=uf(best_k)+ui(best_k);
            moved_in(best_k) = moved_in(best_k)+1;
            changed = changed+1;
        end
    end

    if changed==0, break; end
end
end
