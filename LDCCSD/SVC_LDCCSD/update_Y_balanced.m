function [F,ff,obj_F,changed,converge_f] = update_Y_balanced(U, F, lambda_bal, maxIter, min_size)
% Row-sweep with cluster-balance regularizer.
% U: n×c (你以前传的是 -Q)，F: n×c one-hot
% lambda_bal: 平衡系数（如 1e-3 ~ 1e-2），maxIter: 行扫轮数（2~5足够）
% min_size: 每簇最小样本数（至少=1；可设 floor(0.2*n/c) 之类）

if nargin<3 || isempty(lambda_bal), lambda_bal = 1e-3; end
if nargin<4 || isempty(maxIter),    maxIter    = 5;     end
if nargin<5 || isempty(min_size),   min_size   = 1;     end

[n,c] = size(F);
obj_F = zeros(maxIter+1,1);
changed = zeros(maxIter,1);
converge_f = true;

ff = sum(F,1);               % 各簇大小 (1×c)
uf = sum(U.*F,1);            % 与 U 的对齐项 (1×c)
up = uf./sqrt(ff+eps);       % 目标的“列贡献”
obj_F(1) = sum(up) - lambda_bal * sum((ff - n/c).^2);

incre_F = zeros(1,c);
for it = 1:maxIter
    converged = true;
    for i = 1:n
        ui = U(i,:);                 % 1×c
        id0 = find(F(i,:)==1,1);
        if ff(id0) <= min_size       % 不允许把最小簇再挪空/过小
            continue;
        end

        % 原始目标的增益（与你原函数一致）
        for k = 1:c
            if k == id0
                % leave id0
                incre_raw = uf(k)/sqrt(ff(k)+eps) - (uf(k)-ui(k))/sqrt(ff(k)-1+eps);
            else
                % join k
                incre_raw = (uf(k)+ui(k))/sqrt(ff(k)+1+eps) - uf(k)/sqrt(ff(k)+eps);
            end
            % 平衡正则的增益：-lambda * [ (ff_k+1 - m)^2 - (ff_k-m)^2 + (ff_id0-1 - m)^2 - (ff_id0-m)^2 ]
            % 推导简化 = -2*lambda*(ff_k - ff_id0 + 1)
            incre_bal = -2*lambda_bal * (ff(k) - ff(id0) + 1);
            incre_F(k) = incre_raw + incre_bal;
        end

        [~, id] = max(incre_F);
        if id ~= id0
            converged = false; changed(it) = changed(it)+1;

            % 更新 one-hot
            F(i,id0) = 0; F(i,id) = 1;

            % 即时更新统计量
            ff(id0) = ff(id0) - 1;    ff(id)  = ff(id)  + 1;
            uf(id0) = uf(id0) - ui(id0);
            uf(id)  = uf(id)  + ui(id);
        end
    end

    up = uf./sqrt(ff+eps);
    obj_F(it+1) = sum(up) - lambda_bal * sum((ff - n/c).^2);

    if converged, break; end
    if obj_F(it+1) < obj_F(it) - 1e-12, converge_f = false; end
end
end
