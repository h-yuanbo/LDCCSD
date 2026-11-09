obj_current = compute_spl_objective(X, G, Y, H, nbrIdx, u_new, gamma, tau, K, G_tilde, Y_tilde);
    
    % Try gradient step with backtracking
    for bt = 1:8
        % Standard gradient descent step (minimization)
        u_candidate = u_new - eta * grad;
        
        % Project to feasible region [epsilon, 1-epsilon]
        epsilon = 0.05;  % Avoid extreme values that cause numerical issues
        u_candidate = max(epsilon, min(1-epsilon, u_candidate));
        
        % SPL-specific adjustment: encourage gradual increase when tau increases
        % If most gradients are negative but tau increased, use smaller step
        if mean(grad) < 0 && tau > 0.105  % tau is increasing
            eta = eta * 0.3;  % More conservative steps
        end
        
        % Compute new objective
        obj_new = compute_spl_objective(X, G, Y, H, nbrIdx, u_candidate, gamma, tau, K, G_tilde, Y_tilde);
        
        % Accept if objective decreases (minimization) or if we've tried enough
        if obj_new <= obj_current + 1e-6 || bt >= 8
            u_new = u_candidate;
            if bt < 8 && iter == 1  % Successful step, can increase step size
                etaU = min(etaU * 1.05, 0.8);
            end
            break;
        end
        
        eta = eta * 0.6;  % Reduce step size
    end
    
    % Convergence check
    grad_norm = norm(grad, inf);
    if grad_norm < 1e-5
        break;
    end
    
    % Debug output for first few iterations
    if iter <= 3
        fprintf('  u update iter %d: grad_norm=%.6f, mean(grad)=%.6f, eta=%.6f\n', ...
                iter, grad_norm, mean(grad), eta);
    end
end

% Final SPL regularization: smooth out extreme values
u_mean = mean(u_new);
smoothing_factor = 0.1;
u_new = (1 - smoothing_factor) * u_new + smoothing_factor * u_mean;
u_new = max(0.01, min(0.99, u_new));

end
