function trace_vec = compute_trace_values(X, G, nbrIdx, H, gamma)
    n = size(X, 1);
    trace_vec = zeros(n, 1);
   
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Gi = G(Ni,:);
        Ki = Xi * Xi';
        Mi = Ki + gamma * eye(size(Ki,1));
        HGi = H * Gi;
        trace_vec(i) = trace(HGi' * (Mi \ HGi));
    end
end

