function L_u = assemble_Lu_for_2_3(X, H, nbrIdx, u, gamma)
    n = size(X,1); k = size(H,1);
    L_u = zeros(n,n);
    for i = 1:n
            Ni = nbrIdx(i,:);
            Xi = X(Ni,:);
            Ki = Xi*Xi';
            Mi = Ki + gamma*eye(k); 
            Llocal = H * (Mi \ H);  
            L_u(Ni,Ni) = L_u(Ni,Ni) + u(i) * Llocal; 
    end
    L_u = 0.5*(L_u + L_u');
end
