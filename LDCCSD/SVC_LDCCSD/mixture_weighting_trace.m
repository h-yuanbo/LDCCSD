function u = mixture_weighting_trace(trace, tau, tau_prime,u)
% 混合加权 (Mixture Weighting) - 使用 trace(i), tau, tau_prime
%
% 输入:
%   trace     : [n x 1] 向量，每个样本的平均损失 (已包含C/m系数)
%   tau       : 上阈值 (原来是 1/k)小
%   tau_prime : 下阈值 (原来是 1/k')大
%
% 输出:
%   v         : [n x 1] 权重向量

    n = length(trace);
    % 由 tau, tau_prime 反推 k, k_prime
    k = 1 / tau;
    k_prime = 1 / tau_prime;
    zeta = 1 / (k_prime - k);

    % 遍历每个样本
    for i = 1:n
        if trace(i) <= tau_prime
            u(i) = 1;
        elseif trace(i) >= tau
            u(i) = 0;
        else
            u(i) = (zeta / trace(i)) - k * zeta;
        end
    end
end
