function [Y, objHistory] = solve_Y_entropy_auto_once(L, Y, lambda, e_type)
%
%     min lambda^2 * tr(Y^T L Y) - lambda * entropy([y_1^T y_1, y_2^T y_2, ..., y_c^T y_c])
%
%     min lambda^2 * tr(Y^T L Y) - lambda * entropy([n1, n2, ..., n3])
%
%     min lambda^2 * tr(Y^T L Y) - lambda * entropy([n1/n, n2/n, ..., n3/n])
%
%

nSmp = size(Y, 1);

fLf = sum(Y .* (L * Y))';
ff = sum(Y)';

m_all = vec2ind(Y')';

e = generalized_entropy(ff, e_type);
objHistory = lambda^2 * sum(fLf) - lambda * e;


for i = 1:nSmp
    m = m_all(i);
    if ff(m) == 1
        % avoid generating empty cluster
        continue;
    end

    %*********************************************************************
    % The following matlab code is O(nc)
    % With the loop in n here, it is O(n) actually.
    %*********************************************************************
    Y_A = Y' * L(:, i); 

    fLf_s = fLf + 2 * Y_A + L(i, i); % assign i to all clusters and update
    fLf_s(m) = fLf(m); % cluster m keep the same
    ff_k = ff + 1; % all cluster + 1
    ff_k(m) = ff(m); % cluster m keep the same

    fLf_0 = fLf;
    fLf_0(m) = fLf(m) - 2 * Y_A(m) + L(i, i); % remove i from m
    ff_0 = ff;
    ff_0(m) = ff(m) - 1; % remove i from m
    e_k = generalized_entropy(ff_k, e_type);
    e_0 = generalized_entropy(ff_0, e_type);
    e1 = lambda^2 * fLf_s - lambda * e_k;
    e0 = lambda^2 * fLf_0 - lambda * e_0;
    delta = e1 + sum(e0) - e0;

    [~, p] = min(delta);
    if p ~= m % sample i is moved from cluster m to cluster p
        fLf([m, p]) = [fLf_0(m), fLf_s(p)];
        ff([m, p]) = [ff_0(m), ff_k(p)];
        Y(i, [p, m]) = [1, 0];
        m_all(i) = p;
    end
    e = generalized_entropy(ff, e_type);
    obj = lambda^2 * sum(fLf) - lambda * e;
    objHistory = [objHistory; obj];%#ok
end