function [u, tau,fval] = update_u_fixed_f1(u, X, Y, nbrIdx_nk, H, gamma, tau, options)

% X(d×n), Y(c×n), G(c×n), nbrIdx_nk(n×k), u(n×1)
% tau: 上轮门槛；rho_prev 可忽略，内部给出 rho_t 轨迹
% it/it_max: 外层轮次与总轮次，用于调度 ρ_t

    n = numel(u);

%     % === 1) 计算 b，并做稳健缩放成 b_eff ===
%     b_raw = compute_bvec_only(u, X, Y, G, nbrIdx_nk, H, gamma);   % n×1
%     b_min = min(b_raw);
%     b_shift = b_raw - b_min;                          % ≥0
%     b_scale = median(abs(b_shift)) + eps;             % 稳健尺度
%     b_eff   = b_shift / b_scale;
% 
%     % === 2) 设定当轮预算 ρ_t：缓慢升到 ~1 ===
%     rho_start = 0.10; 
%     rho_end   = 0.75;                 % 不到 1，避免数值抖动
%     ramp = (it/it_max)^1.5;  % 0→1
%     rho_t     = rho_start + (rho_end - rho_start) * ramp;
% 
%     % === 3) 自然增大的 τ（分位 + 裕量），单调不减 ===
%     kappa = 0.05;                     % 裕量系数（0.15~0.5 可试）
%     tau_tgt = quantile(b_eff, rho_t) + kappa * std(b_eff);
%     tau_new = (1 - beta) * tau + beta * tau_tgt;
%     tau     = max(tau, tau_new);      % 保证单调不降
%     tau     = max(tau, 1e-8);

    % === 4) 用 fmincon 解带预算的子问题（自然推进，不硬裁剪）===
%     A = ones(1,n); 
%     b = rho_t * n;                    % sum(u) ≤ ρ_t·n
    lb = zeros(n,1); 
    ub = ones(n,1);

    fun = @(uu) obj_grad_u_3_3_f1(uu, X, Y, nbrIdx_nk, H, gamma, tau);
    [u,fval]   = fmincon(fun, u, [],[], [], [], lb, ub, [], options);

    % === 5) 汇总信息 ===
   
end
