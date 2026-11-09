function test_obj_grad_u()
    rng(42); % 固定随机种子，方便复现

    % 参数设置
    n = 15; d = 5; c = 4; k = 3; gamma = 0.1; tau = 0.5;
    X = randn(d, n);           % 数据矩阵 d×n
    G = orth(randn(n, c));     % 嵌入 n×c
    Y = zeros(n, c);           % 离散标签 (这里随便 one-hot)
    H = eye(k) - (1/k)*ones(k);     
    for i=1:n, Y(i, randi(c))=1; end

    % 邻居索引 O_i
    nbrIdx = zeros(n, k);
    for i=1:n
        nbrIdx(i,:) = randperm(n, k);  % 随机邻居（可以替换为 knn）
    end

    % 初始 u
    u = rand(n,1);

    % --- 调用你的函数 ---
    [f_fun, g_fun] = obj_grad_u_comparison(X', Y, G, u, tau, gamma, H, nbrIdx, 0.0001);
    % 初始化
    % --- 手算版本 ---
    f_manual = 0; g_manual = zeros(n,1);
    for i=1:n
        % 构造 M^i
        Xi = X(:, nbrIdx(i,:));
        Gi = G(nbrIdx(i,:),:);
        Yi = Y(nbrIdx(i,:),:);
        ui = u(nbrIdx(i,:));

        M = diag(ui)* (Xi'*Xi) * diag(ui) + gamma*eye(k);
        Minv = inv(M);

        % a_i
        Ghat = Gi';   % c×k
        Yhat = Yi;    % k×c
        a_i = trace(Ghat * Minv * Yhat);

        % f_i
        f_manual = f_manual + u(i)*a_i - tau*u(i);

        % 梯度: df/du_i
        df = a_i - tau;
        for o = 1:k
            idx = nbrIdx(i,o); % 邻居 id
            uo = u(idx);
            Xo = X(:, nbrIdx(idx,:));
            Go = G(nbrIdx(idx,:),:);
            Yo = Y(nbrIdx(idx,:),:);
            uo_vec = u(nbrIdx(idx,:));

            Mo = diag(uo_vec)*(Xo'*Xo)*diag(uo_vec) + gamma*eye(k);
            Minv_o = inv(Mo);
            Ghat_o = Go'; Yhat_o = Yo;

            % Hadamard 项
            term = Minv_o*(Yhat_o*Ghat_o + Ghat_o'*Yhat_o')*Minv_o;
            Had = (Xo'*Xo) .* term;

            e_v = zeros(k,1); e_v(o)=1; % 指示向量
            df = df - uo * (uo_vec' * (Had * uo_vec));
        end
        g_manual(i) = df;
    end

    % --- 对比 ---
    fprintf('f_fun = %.6f, f_manual = %.6f, diff = %.2e\n', f_fun, f_manual, abs(f_fun-f_manual));
    fprintf('||g_fun - g_manual|| = %.2e\n', norm(g_fun-g_manual));
    disp([g_fun g_manual]);
end
