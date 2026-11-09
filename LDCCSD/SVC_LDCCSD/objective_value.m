function obj = objective_value(u , X, Y, G, nbrIdx, H, gamma, tau)
    % obj(u,G,Y) = sum_i u_i * tr( G~^i M_i^{-1} Y~^i ) - tau * sum_i u_i
    %               with correlated neighborhoods (through M_i)
    n = size(X,1); k = size(nbrIdx,2); c = size(Y,2);
    term = 0;
    for i = 1:n%慢
        Ni = nbrIdx(i,:);
        Xi = X(Ni,:);
        Ki = Xi * Xi';
        ui = u(Ni);
        Du = diag(ui);
        I=eye(k);
        Mi = Du * Ki * Du + gamma*I;

        Gi = G(Ni,:);
        Yi = Y(Ni,:);
        Gtil = (Gi') * H;                         % c x k
        %YtY_i = (Yi') * Yi;
        %[Vd, Dd] = eig(YtY_i + 1e-12*eye(c));
        invsqrtYtY=diag(1./sqrt(sum(Y,1)+eps));
        Ytil = H * Yi * invsqrtYtY;             % k x c
        tmp=Gtil/Mi;
        term=term+u(i)*sum(sum(tmp.*Ytil));
        % = term + u(i) * trace( Gtil / Mi * Ytil );
    end
    obj = term - tau * sum(u);
end