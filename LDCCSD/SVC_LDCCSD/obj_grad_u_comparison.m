function [obj, grad] = obj_grad_u_comparison(X, Y, G, u, tau, gamma, H, nbrIdx, eps_eye)
    % 初始化
    n = size(X, 1);           % 节点数
    k = size(nbrIdx, 2);      % 每个节点的邻居数量
    c = size(Y, 2);           % 类别数
    grad = zeros(n, 1);       % 梯度初始化

    % 计算目标函数值和梯度
    obj = 0;
for o = 1:n
            No = nbrIdx(o,:);                  % k indices
            Xo = X(No,:);                      % k x d
            Ko = Xo * Xo.';                    % k x k
            uo = u(No);                        % k x 1
            Du = diag(uo);
            Mo = Du * Ko * Du + gamma*eye(k);

            Go = G(No,:);                      % k x c
            Yo = Y(No,:);
            % Gtil = (G^o)' H  (c x k),  Ytil = H Y^o (Y^o'Y^o)^(-1/2)  (k x c)
            Gtil = Go.' * H;                   % c x k

            YtY = Yo.' * Yo;
            % (Y^T Y)^(-1/2) with regularization
            [V,D] = eig(YtY + eps_eye*eye(c));
            invsq = V * diag(1./sqrt(max(diag(D),eps_eye))) * V.';
            Ytil  = H * Yo * invsq;            % k x c

            % a_o term = tr( G~^o M^{-1} Y~^o )
            % (Gtil / Mo) is c x k ; times (Ytil) k x c  -> c x c ; take trace
            a(o) = trace( (Gtil / Mo) * Ytil );

            % coupling term:
            % T = M^{-1}(Y~G~ + (Y~G~)^T)M^{-1}
            YG = Ytil * Gtil;                  % k x k
            T  = (Mo \ YG) / Mo + (Mo \ YG.') / Mo;
            gvec = - (Ko .* T) * uo;           % k x 1

            % accumulate to indices in No
            % grad(i) += u(o) * gvec(v)
            % (we keep the loop for clarity; k is small)
            for v = 1:k
                i = No(v);
                grad(i) = grad(i) + u(o) * gvec(v);
            end
        end

        % total gradient: df/du_i = a_i - tau + coupling
        grad = a - tau + grad;
    % 计算目标函数
    obj = sum(a) - tau * sum(u); % 目标函数值
end
