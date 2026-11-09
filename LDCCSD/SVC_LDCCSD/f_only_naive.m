function fval = f_only_naive(u, X, Y, G, nbrIdx, H, gamma, tau)
% f(u) = sum_o [ u_o * b_o(u^o) ] - tau * sum_i u_i
% b_o = tr( G~^o * (M^o)^{-1} * Y~^o ),
% M^o = diag(u^o) K^o diag(u^o) + gamma I_k,
% K^o = (X^o)' X^o,
% Y~^o = H * (Y^o ./ ||列||)^T,  G~^o = (H (G^o)^T)^T
[~, n] = size(X);
k = size(nbrIdx,1);
fsum = 0;
for o = 1:n
    idx = nbrIdx(:,o);
    Xo = X(:,idx);                       % d×k
    Ko = Xo' * Xo;                       % k×k
    uo = u(idx);                         % k×1
    Mo = diag(uo) * Ko * diag(uo) + gamma*eye(k);

    Yo = Y(:,idx);                       % c×k
    cols_norm = sqrt(sum(Yo.^2,1)) + eps;
    Ynorm = Yo ./ cols_norm;             % c×k
    Ytilde = H * (Ynorm.');              % k×c

    Go = G(:,idx);                       % c×k
    Gtilde = (H * Go')';                 % c×k

    MinvY = Mo \ Ytilde;                 % k×c
    Fo = Gtilde * MinvY;                 % c×c
    bo = trace(Fo);

    fsum = fsum + u(o) * bo;
end
fval = fsum - tau * sum(u);
end

