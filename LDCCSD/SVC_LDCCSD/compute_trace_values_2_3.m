function trace_vec = compute_trace_values_2_3(X, Y, nbrIdx, H, gamma)
    n = size(X, 1);
    trace_vec = zeros(n, 1);
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);        % k×d
        Yi = Y(Ni,:);        % k×c
        
        Ki = Xi * Xi';       % k×k
        Mi = Ki + gamma * eye(size(Ki,1));  % k×k
        
        % 双重Y归一化
        YtY = Yi' * Yi;      % c×c
        invsqrt_YtY = diag(1./sqrt(diag(YtY) + eps));  % c×c
        
        HYi = H * Yi;        % k×c
        % 正确的计算: (Y^{iT}Y^i)^{-1/2} Y^{iT} H^i (Mi)^{-1} H^i Y^i (Y^{iT}Y^i)^{-1/2}
        temp = HYi' * (Mi \ HYi);  % c×k × k×c = c×c
        trace_vec(i) = trace(invsqrt_YtY * temp * invsqrt_YtY);
    end
end
