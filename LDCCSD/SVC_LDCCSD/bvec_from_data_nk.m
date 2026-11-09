function b_vec = bvec_from_data_nk(X, Y, G, nbrIdx_nk, H, gamma, u)
% 与 obj_grad_u2 中 b_vec 定义一致；只算 b，不算梯度。
    [d, n] = size(X);
    c      = size(G,1);
    k      = size(nbrIdx_nk, 2);          % nbrIdx: n×k

    nbrIdx = nbrIdx_nk.';                  % 转成 k×n 以复用张量套路
    cols   = nbrIdx(:);                    % (k*n)×1

    % -- K^o --
    X_tensor = reshape(X(:, cols), d, k, n);                       
    K_tensor = pagemtimes(permute(X_tensor,[2 1 3]), X_tensor);   

    % -- Y~^o, G~^o --
    H_tensor = repmat(H, 1, 1, n);                                

    Y_tensor  = reshape(Y(:, cols), c, k, n);                      
    col_norms = sqrt(sum(Y_tensor.^2, 1)) + eps;                   
    Y_norm    = Y_tensor ./ col_norms;                             
    Ytilde    = pagemtimes(H_tensor, permute(Y_norm,[2 1 3]));     

    G_tensor  = reshape(G(:, cols), c, k, n);                      
    G_kc      = permute(G_tensor,[2 1 3]);                         
    Ghat      = permute(pagemtimes(H_tensor, G_kc), [2 1 3]);      

    % -- M^o --
    u_neighbors = reshape(u(nbrIdx), k, n);                        
    U_outer     = pagemtimes(permute(u_neighbors,[1 3 2]), ...
                             permute(u_neighbors,[3 1 2]));        
    M_tensor    = U_outer .* K_tensor + gamma * repmat(eye(k),1,1,n);
    Minv_tensor = pageinv(M_tensor);

    % -- b_o = tr(G~^o * M^{-1}Y~^o) --
    A_tensor = pagemtimes(Minv_tensor, Ytilde);                    
    F_tensor = pagemtimes(Ghat, A_tensor);                         

    I_c_pages = repmat(eye(size(G,1)), 1, 1, n);
    b_vec = squeeze(sum(sum(F_tensor .* I_c_pages, 1), 2));        % n×1
end