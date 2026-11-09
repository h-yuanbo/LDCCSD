function [u, tau,fval] = update_u_fixed(u, X, Y, nbrIdx_nk, H, gamma, tau, options)

% X(d×n), Y(c×n), G(c×n), nbrIdx_nk(n×k), u(n×1)

    n = numel(u);
    lb = zeros(n,1); 
    ub = ones(n,1);
    fun = @(uu) obj_grad_u_3_3(uu, X, Y, nbrIdx_nk, H, gamma, tau);
    [u,fval]= fmincon(fun, u, [],[], [], [], lb, ub, [], options);
   
end
