function [Y] = update_Y(Q, Y)
% Update Y according to paper Eq.(...) using O(nc) method
% Q: n x c matrix  (Q = L_u * G in paper)
% Y: n x c one-hot cluster assignment matrix
% ny: 1 x c vector, cluster sizes
% qy: 1 x c vector, q_r^T y_r
% obj_Y: objective function values per iteration
% changed: number of changed assignments per iteration
% converge_y: flag for convergence

[n, c] = size(Y);
obj_Y = zeros(11, 1);  
changed = zeros(10, 1);
converge_y = true;

% ---- Precompute auxiliary variables (Eq. paper)
ny = sum(Y, 1);                  % y_r^T y_r
qy = sum(Q .* Y, 1);              % q_r^T y_r

% Initial objective value (sum over r)
obj_Y(1) = sum(qy ./ sqrt(ny + eps));

% Iterative update (max 10 passes)
for iter = 1:10
    converged = true;
    for j = 1:n
        qj = Q(j, :);
        rho = find(Y(j, :) == 1); % current cluster index
        
        % ---- L(e^0): remove from current cluster rho (Eq. step_1)
        L0 = (qy(rho) - qj(rho)) / sqrt(ny(rho) - 1 + eps);
        
        % ---- Try all possible target clusters
        gain = -inf(1, c);
        for k = 1:c
            if k == rho
                gain(k) = L0;
            else
                % ---- L(e^k): add to cluster k (Eq. step_2)
                gain(k) = (qy(k) + qj(k)) / sqrt(ny(k) + 1 + eps);
            end
        end
        
        % ---- Choose best cluster
        [~, new_rho] = max(gain);
        if new_rho ~= rho
            converged = false;
            changed(iter) = changed(iter) + 1;
            
            % ---- Update Y
            Y(j, rho) = 0; Y(j, new_rho) = 1;
            
            % ---- Update auxiliary variables in O(c)
            ny(rho) = ny(rho) - 1;
            ny(new_rho) = ny(new_rho) + 1;
            qy(rho) = qy(rho) - qj(rho);
            qy(new_rho) = qy(new_rho) + qj(new_rho);
        end
    end
    
    % Check convergence
    if converged
        break;
    end
    
    % Compute new objective
    obj_Y(iter+1) = sum(qy ./ sqrt(ny + eps));
    
    % Check if objective decreased
    if obj_Y(iter+1) - obj_Y(iter) < 0
        converge_y = false;
    end
end
end
