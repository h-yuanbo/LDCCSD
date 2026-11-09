function [u_new, tau, fval] = update_u_fixed_v2sim(u, X, Y, G, nbrIdx_nk, H, gamma, tau, beta, options, it, it_max)
    n  = numel(u);
    lb = zeros(n,1); 
    ub = ones(n,1);

    % —— 自适应 τ：按 b 的分位数设置 —— 
    %b_vec = compute_bvec_only(u, X, Y, G, nbrIdx_nk, H, gamma);
    
   
   
    fun = @(uu) obj_grad_u2(uu, X, Y, G, nbrIdx_nk, H, gamma, tau);
    [u_new, fval] = fmincon(fun, u, [], [], [], [], lb, ub, [], options);

    %if norm(u - u_new) < 1e-6
        %fprintf('[WARNING] u 变化极小，可能被 τ/约束卡死\n');
    %end
end
