function [f, g, aux] = obj_grad_u2_fixed(u, cache, tau)
K_tensor = cache.K_tensor; Ytilde = cache.Ytilde; Ghat = cache.Ghat;
G_kc = cache.G_kc; nbrIdx = cache.nbrIdx; gamma = cache.gamma;

[~, ~, n] = size(K_tensor);
k = size(K_tensor,1); c = size(Ghat,1);

u_neighbors = reshape(u(nbrIdx), k, n);
U_outer = pagemtimes(permute(u_neighbors,[1 3 2]), permute(u_neighbors,[3 1 2]));
M_tensor = U_outer .* K_tensor + gamma * repmat(eye(k),1,1,n);

A = zeros(k,c,n);
R_all = zeros(k,k,n);
for t = 1:n
    M = M_tensor(:,:,t);
    [R,p] = chol(M);
    if p>0, M = M + (1e-10)*eye(k); R = chol(M); end
    R_all(:,:,t) = R;
    A(:,:,t) = R \ (R' \ Ytilde(:,:,t));
end

b_vec = zeros(n,1);
for t = 1:n
    b_vec(t) = trace(Ghat(:,:,t) * A(:,:,t));
end

% —— 非负重标化：避免 u 一步饱和
b_min = min(b_vec);
b_cent = b_vec - b_min;                  % >= 0
scale  = median(abs(b_cent)) + eps;      % 稳健尺度
b_eff  = b_cent / scale;                 % 用于目标与 g 的常量项

f = sum(u .* b_eff) - tau * sum(u);
g = b_eff - tau*ones(n,1);

db_duo = zeros(k,n);
for t = 1:n
    P  = Ytilde(:,:,t) * Ghat(:,:,t);
    S0 = P + P.';
    R = R_all(:,:,t);
    W = R \ (R' \ S0);
    S = R \ (R' \ W.'); S = S.';
    Had = K_tensor(:,:,t) .* S;
    db_duo(:,t) = -(Had * u_neighbors(:,t))/scale; % 注意除以 scale
end
rows = nbrIdx(:);
vals = reshape(db_duo, [],1) .* reshape(repmat(u.',k,1),[],1);
g = g + accumarray(rows, vals, [n,1], @sum, 0);

aux = struct('b_vec',b_vec,'b_min',b_min,'scale',scale,'b_eff',b_eff);
end
