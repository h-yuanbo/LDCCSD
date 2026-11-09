function L_u = assemble_Lu_basic(X, H, nbrIdx, gamma)
    % Basic LDC without u weighting (for objectives 1.x)
    n = size(X,1); k = size(H,1);
    L_u = zeros(n,n);
    for i = 1:n
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Ki = Xi*Xi';
        Mi = Ki + gamma*eye(k);
        Llocal = H / Mi * H;
        L_u(Ni,Ni) = L_u(Ni,Ni) + Llocal; % No u_i weighting
    end
    L_u = 0.5*(L_u + L_u');
end


