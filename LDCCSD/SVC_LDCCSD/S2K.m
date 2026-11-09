function K = S2K(S)
n = size(S, 1);
% Symmetrize
S = max(S, S');

% Degree matrix
d = sum(S,2);
D_inv_sqrt = spdiags(1 ./ sqrt(d + eps), 0, n, n);

% Normalized kernel
K = D_inv_sqrt * S * D_inv_sqrt;
K = (K + speye(n)) / 2;
end