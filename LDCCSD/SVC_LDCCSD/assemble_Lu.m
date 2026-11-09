function L_u = assemble_Lu(X, H, nbrIdx, u, gamma)
    % Assemble the global L_u = sum_i u_i * S^i * L_{u^i} * (S^i)'
    n = size(X,1); k = size(H,1);
    L_u = zeros(n,n);
    % Precompute neighbor features
    for i = 1:n
        Ni = nbrIdx(i,:);                          % indices of neighbors of i
        Xi = X(Ni,:);                              % k x d (MATLAB row-major)
        Ki = Xi*Xi';                               % k x k  (X^{iT} X^i)
        ui = u(Ni);                                % u^i (k x 1)
        Du = diag(ui);
        Mi = Du * Ki * Du + gamma*eye(k);
        Llocal = H / Mi * H;                       % H * inv(Mi) * H  (inv via /)
        % scatter into global via selection S^i (implicit): L_u += u_i * S^i Llocal (S^i)'
        L_u(Ni,Ni) = L_u(Ni,Ni) + u(i) * Llocal;   % only the (Ni,Ni) block
    end
    % Ensure symmetry (numerical safety)
    L_u = 0.5*(L_u + L_u');
end