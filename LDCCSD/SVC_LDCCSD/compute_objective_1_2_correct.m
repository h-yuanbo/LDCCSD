function obj_val = compute_objective_1_2_correct(X, G, Y, nbrIdx, H, gamma, epsY)
    % 正确计算1.2目标函数值
    obj_val = 0;
    k = size(H, 1);
    
    for i = 1:size(X, 1)
        idx = nbrIdx(i, :);
        Xi = X(idx, :);
        Ki = Xi * Xi';
        Ai = inv(Ki + gamma * eye(k));
        
        Yi = Y(idx, :);
        YtY = Yi' * Yi + epsY * eye(size(Yi, 2));
        YtY_inv_sqrt = YtY^(-0.5);
        
        % 如果数值不稳定，跳过这一项
        if any(isnan(YtY_inv_sqrt(:))) || any(isinf(YtY_inv_sqrt(:)))
            continue;
        end
        
        % tr[G^{iT} H^i A^i H^i Y^i (Y^{iT}Y^i)^{-1/2}]
        term = G(idx, :)' * H * Ai * H * Yi * YtY_inv_sqrt;
        obj_val = obj_val + trace(term);
    end
end