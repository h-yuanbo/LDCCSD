function L_u = assemble_Lu_v2(X, H, nbrIdx, u, gamma)
    n = size(X,1); k = size(H,1);
    L_u = zeros(n,n);

    % 建议：外层只用温和权重（别再乘 u(i)），或者先归一化到 sum w_i = n
    w = ones(n,1);             % 或者 w = u / mean(u);
    % w = u / (mean(u)+eps);   % 如果一定要用 u，当作温和 reweight

    for i = 1:n
        Ni = nbrIdx(i,:);                  % k 索引
        Xi = X(Ni,:);                      % k×d
        Ki = Xi*Xi.';                      % k×k
        ui = u(Ni);                        % k×1
        Du = diag(ui);

        % --- 正则的“标定”，避免不同块量级差太大 ---
        % 用 Ki 的迹/谱半径做尺度：
        trK = trace(Ki)/k;                 % 平均特征能量
        % 也可用 max(eig(Ki)) 但更贵
        Mi = Du * Ki * Du + (gamma * trK + 1e-8) * eye(k);

        % 局部算子
        Lloc = H / Mi * H;                 % H * inv(Mi) * H

        % --- 块归一化：压平不同块的量级 ---
        % 任选其一：Fro 或 trace 归一化
        s = norm(Lloc,'fro') + 1e-12;
        Lloc = Lloc / s;

        % 散射到全局（只用 w(i) 或者 1；避免再次叠加 u(i)）
        L_u(Ni,Ni) = L_u(Ni,Ni) + w(i) * Lloc;
    end

    % 数值对称化
    L_u = 0.5 * (L_u + L_u.');
end
