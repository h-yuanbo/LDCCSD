function e = generalized_entropy(p, e_type)
% generalized partition entropy
% c = length(p);
% [1] On defining partition entropy by inequalities.
% IEEE Transactions on Information Theory, Ping Luo.
% 
p = (p + eps) ./ sum(p); % avoid zero
switch e_type
    case 1
        e = -sum( p .* log(p));
    case 2
        e = -sum( p .* log2(p));
    case 3
        e = sum( p .* (exp( 0.5 - p) + 0));
    case 4
        e = sum( p .* (exp( 1 - p) + 0));
    case 5
        e = sum( p .* (exp( 2 - p) + 0));
    case 6
        e = sum( p .* (exp( 1 - p) + 2));
    case 7
        e = sum(p .* (1-p));        
    case 8
        e = 1 - sum(p.^1.5);
    case 9
        e = 1 - sum(p.^2);
    case 10
        e = 1 - sum(p.^4);
    case 11
        e = sum(p.^0.1) - 1;
    case 12
        e = sum(p.^0.5) - 1;
    case 13
        e = sum(p.^0.9) - 1;
    case 14
        e = 1 - max(p);
    otherwise
        e = 0;
end

end