function HLP = compute_smooth_P_by_HK(P, eta)
[~, m] = size(P);

PTP = P' * P; % m * m
Lm = eye(m) - PTP;
HLm = expm(-eta * Lm);
HLP = P * HLm; % n * m * (m * m)
end