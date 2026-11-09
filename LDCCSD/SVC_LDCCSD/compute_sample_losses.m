function losses = compute_sample_losses(X, G, Y, H, nbrIdx, u, gamma)
    % compute per-center loss a(o) = trace( Gtil / Mo * Ytil )
    n = size(X,1); k = size(nbrIdx,2);
    losses = zeros(n,1);
    for o = 1:n
        neigh = nbrIdx(o,:);
        Xo = X(neigh, :);
        Ko = Xo * Xo';
        uo = u(neigh);
        Du = diag(uo);
        Mo = Du * Ko + gamma * eye(k);
        Go = G(neigh, :);
        Yo = Y(neigh, :);
        Gtil = Go' * H;
        YtY_o = Yo' * Yo;
        [Vd, Dd] = eig(YtY_o + 1e-12*eye(size(YtY_o)));
        invsqrtYtY_o = Vd * diag(1./sqrt(max(diag(Dd),1e-12))) * Vd';
        Ytil = H * Yo * invsqrtYtY_o;
        % a(o)
        losses(o) = trace( Gtil / Mo * Ytil );
    end
end