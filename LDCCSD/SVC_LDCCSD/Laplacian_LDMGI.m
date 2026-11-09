function L = Laplacian_LDMGI(X, para)
% each column is a data

[D, n] = size(X);

if isfield(para, 'k')
    k = para.k;
else
    k = 10;
end;
if isfield(para, 'gamma')
    gamma = para.gamma;
else
    gamma = 10000;
end;

Lc = eye(k) - 1/k*ones(k);
A = spalloc(n*k,n*k,5*n*k);
S = spalloc(n,n*k,5*n*k);
for i = 1:n
    dis = bsxfun(@minus, X(:,i), X);
    dis = sum(dis.*dis);
    [dumb, nnidx] = sort(dis);
    Xi = X(:,nnidx(1:k));
    Xi = Xi*Lc;
    if D > k
        Ai = inv(gamma*eye(k) + Xi'*Xi);
        Ai = Lc*Ai*Lc;
     else
          Ai = Lc - gamma*Xi'*inv(eye(D) + gamma*Xi*Xi')*Xi;
    end;
    lidx = (i-1)*k+1:(i-1)*k+k;
    A(lidx, lidx) = Ai;
    S(nnidx(1:k),lidx) = eye(k);
end;
    
L = S*A*S';




