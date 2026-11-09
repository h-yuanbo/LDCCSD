function fval = compute_objective_1_2(X, G, Y, nbrIdx, H, gamma)
    n = size(X, 1);
    fval = 0;
    
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Gi = G(Ni,:);
        Yi = Y(Ni,:);
        
        Ki = Xi * Xi';
        Mi = Ki + gamma * eye(size(Ki,1));
        
        % Y归一化项
        YtY = Yi' * Yi;
        invsqrt_YtY = diag(1./sqrt(diag(YtY) + eps));
        Yi_norm = Yi * invsqrt_YtY;
        
        HYi = H * Yi_norm;
        fval = fval + trace(Gi' * HYi * (HYi' * (Mi \ HYi)));
    end
end