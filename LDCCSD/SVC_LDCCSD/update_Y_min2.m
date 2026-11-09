function [F, obj_F] = update_Y_min2(U, F, innerIterY)
[n,c] = size(F);

% 添加调试信息
fprintf('update_Y_min开始: n=%d, c=%d, innerIterY=%d\n', n, c, innerIterY);

obj_F = zeros(11,1);           
ff = sum(F);                        
uf = sum(U.*F);                     

% 检查初始状态
fprintf('初始聚类大小: %s\n', mat2str(ff));
fprintf('是否有空聚类: %d\n', sum(ff == 0));

up = zeros(1,c);
for cc = 1:c                        
    up(cc) = uf(cc)./sqrt(ff(cc)+eps);
end
obj_F(1) = sum(up);                 

changed = zeros(innerIterY,1);
incre_F = zeros(1,c);
total_changes = 0;

for iterf = 1:innerIterY                    
    converged = true;
    iter_changes = 0;
    
    for i = 1:n
        ui = U(i,:);
        [~,id0] = find(F(i,:)==1);
        
        % 记录跳过的情况
        if ff(id0) <= 1 && sum(ff > 0) <= c
            % fprintf('样本%d被跳过，当前聚类%d大小为%d\n', i, id0, ff(id0));
            continue;
        end
        
        for k = 1:c                          
            if k == id0
                incre_F(k) = uf(k)/sqrt(ff(k)+eps) - (uf(k) - ui(k))/sqrt(max(ff(k)-1, eps));
            else
                incre_F(k) = (uf(k)+ui(k))/sqrt(ff(k)+1+eps) - uf(k)/sqrt(ff(k)+eps);
            end
        end

        [obj_delta, id] = min(incre_F);
        if id ~= id0 && obj_delta < 0                          
            converged = false;               
            iter_changes = iter_changes + 1;
            total_changes = total_changes + 1;
            
            F(i,id0) = 0; F(i,id) = 1;
            ff(id0) = ff(id0) - 1;           
            ff(id)  = ff(id) + 1;            
            uf(id0) = uf(id0) - ui(id0);
            uf(id)  = uf(id) + ui(id);
        end
    end
    
    fprintf('  内循环%d: 改变了%d个样本\n', iterf, iter_changes);
    
    if converged
        fprintf('  第%d次内循环收敛\n', iterf);
        break;
    end
    
    for cc = 1:c
        up(cc) = uf(cc)/sqrt(ff(cc)+eps);
    end
    obj_F(iterf+1) = sum(up);
end

fprintf('update_Y_min结束: 总共改变了%d个样本\n', total_changes);
fprintf('最终聚类大小: %s\n', mat2str(ff));

obj_F(iterf+1:end) = [];
end
