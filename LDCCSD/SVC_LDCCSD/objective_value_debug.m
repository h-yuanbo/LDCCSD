function [obj, data_term, reg_term] = objective_value_debug(X, G, Y, H, nbrIdx, u, gamma, tau)
% ------------------------------------------------------------
% Debug version of objective_value
%   输出:
%     obj       : 总目标函数值
%     data_term : 数据项 (和 u_i * trace(...) 有关)
%     reg_term  : 正则/惩罚项 tau * sum(u)
% ------------------------------------------------------------

n = size(X,1);
c = size(G,2);

data_term = 0;

for i = 1:n
    Xi = X(nbrIdx(i,:), :);     % k x d
    Gi = G(nbrIdx(i,:), :);     % k x c
    Yi = Y(nbrIdx(i,:), :);     % k x c
    ui = u(nbrIdx(i,:));        % k x 1
    
    Ki = Xi*Xi';                
    Mi = diag(ui)*Ki*diag(ui) + gamma*eye(size(Ki));
    
    % --- 数值稳定处理 (Yi'Yi)^(-0.5) ---
    YtY = Yi' * Yi;
    [V,D] = eig(YtY + 1e-12*eye(c));
    D = max(diag(D),1e-12);
    invsqrtYtY = V * diag(1./sqrt(D)) * V';
    
    Gtil = Gi' * H;                % c x k
    Ytil = H * Yi * invsqrtYtY;    % c x k
    
    term = trace(Gtil * (Mi \ Ytil));
    
    % ---- 注意这里乘 u(i) ----
    data_term = data_term + u(i) * term;
end

reg_term = tau * sum(u);

obj = data_term - reg_term;
end
