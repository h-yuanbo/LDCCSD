function fval = compute_objective_1_3(X, Y, nbrIdx, H, gamma)
    n = size(X, 1);
    fval = 0;
    
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Yi = Y(Ni,:);
        
        Ki = Xi * Xi';
        Mi = Ki + gamma * eye(size(Ki,1));
        
        % 双重Y归一化
        YtY = Yi' * Yi;
        invsqrt_YtY = diag(1./sqrt(diag(YtY) + eps));
        
        HYi = H * Yi;
        temp = HYi' * (Mi \ HYi);
        fval = fval + trace(invsqrt_YtY * temp * invsqrt_YtY);
    end
end