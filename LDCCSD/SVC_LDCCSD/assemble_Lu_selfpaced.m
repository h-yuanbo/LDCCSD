function L_u = assemble_Lu_selfpaced(X, H, nbrIdx, u, gamma)
    % Self-paced without correlation (for objectives 2.x)
    n = size(X,1); k = size(H,1);
    L_u = zeros(n,n);
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Ki = Xi*Xi';
        Mi = Ki + gamma*eye(k); % No diag(u^i) weighting in Ki
        Llocal = H / Mi * H;
        L_u(Ni,Ni) = L_u(Ni,Ni) + u(i) * Llocal; % u_i weighting only
    end
    L_u = 0.5*(L_u + L_u');
end