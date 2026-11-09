function [F L eigval] = LGC(X, para)
% ranking by Laplacian
% each column is a data

if isfield(para, 'k')
    para.k = para.k;
else
    para.k = 10;
end
if isfield(para, 'gamma')
    para.gamma = para.gamma;
else
    para.gamma = 10000;
end
L = Laplacian_LDMGI(X,para);
M = (L+L')/2; M = full(M);
[v,eigval] = eig(M);
[eigval, idx] = sort(diag(eigval));
F = v(:, idx(2:para.nCluster+1));







