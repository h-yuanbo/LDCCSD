function L_u = assemble_Lu_standard(X, H, nbrIdx, u, gamma)
    [n, ~] = size(X);
    k = size(nbrIdx, 2);
    L_u = zeros(n, n);
    
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Ki = Xi * Xi';
        Mi = Ki + gamma * eye(k);
        HMH = H * (Mi \ H);
        L_u(Ni, Ni) = L_u(Ni, Ni) + u(i) * HMH;
    end
end

