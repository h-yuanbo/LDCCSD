function H = compute_entropy(p_k, entropy_type)
% 检查输入是否有效
if ~isnumeric(p_k) || length(p_k) < 1
    error('p_k must be a numeric vector with at least one element.');
end
if ~ischar(entropy_type) || isempty(entropy_type)
    error('entropy_type must be a string.');
end

% 计算熵度
switch entropy_type
    case 'Sqrt'
        H = 1./sqrt(p_k);
    case 'Sqrt_inverse'
        H = sqrt(p_k);
    case 'Shannon'
        H = -p_k .* log(p_k + eps); % 加上 eps 防止 log(0)
    case 'Shannon2'
        H = p_k .* log(p_k + eps); % 加上 eps 防止 log(0)
    case 'Shannon_inverse'
        H = 1./(-p_k .* log(p_k + eps));
    case 'Gini'
        H = 1 - p_k.^2;
    case 'Gini1'
        H = p_k.^2;
    case 'Gini2'
        H = 1./(p_k.^2);
    case 'Minimum'
        H = 1 - p_k;
    case 'Uniformity'
        uniform_p = 1/length(p_k);
        H = min(p_k, uniform_p);
    case 'Exponential'
        mu = 0.5; % 仅在此处定义 mu
        H = 1 - exp(-mu * p_k);
    case 'Exponential2'
        mu = 0.5; % 仅在此处定义 mu
        H = exp(-mu * p_k);
    case 'Exponential3'
        mu = 0.5; % 仅在此处定义 mu
        H = -exp(-mu * p_k);
    case 'Exponential_u5'
        mu = 5; % 仅在此处定义 mu
        H = exp(-mu * p_k);
    case 'Exponential_u6'
        mu = 6; % 仅在此处定义 mu
        H = exp(-mu * p_k);
    case 'Exponential_u10'
        mu = 10; % 仅在此处定义 mu
        H = exp(-mu * p_k);
    case 'Exponential_inverse'
        mu = 0.5; % 仅在此处定义 mu
        H = 1./(1 - exp(-mu * p_k));
    otherwise
        error('Unknown entropy type: %s', entropy_type);
end
end