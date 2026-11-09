function [G,M]= update_G(L_u, Y)
    % Solve: min_G tr(G^T L_u Y (Y^T Y)^(-1/2)) s.t. G^T G = I_c
    % Equivalent to orthogonal Procrustes on M = L_u * Y * (Y^T Y)^(-1/2)
    [n, c] = size(Y);
    %YtY = Y' * Y;
    
    invsqrtYtY=diag(1./sqrt(sum(Y,1)+eps));
    
    M = L_u * Y * invsqrtYtY;
    %M = L_u * Y ;
    [U,~,V] = svd(M,'econ');
    G = -U*V';
end