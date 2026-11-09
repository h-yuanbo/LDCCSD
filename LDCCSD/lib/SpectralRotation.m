function [F_rot, obj, Fq] = SpectralRotation(F)
    % Spectral Rotation for matrix F
    % F: The feature matrix to be rotated (each column is a data point)
    % F_rot: Rotated feature matrix
    % obj: Objective function value for each rotation
    % Fq: Modified feature matrix

    [n, m] = size(F);  % n: number of data points, m: number of features
    F_rot = F;  % Initialize rotated features as original
    obj = zeros(1, m);  % Initialize the objective function values for each rotation
    Fq = zeros(n, m);  % Initialize the modified feature matrix
    
    % Perform spectral rotation for each column (data point)
    for i = 1:m
        % Singular Value Decomposition (SVD) for column i of F
        [U, S, V] = svd(F(:, i), 'econ');
        
        % Perform the rotation step (here we apply a basic rotation using V)
        % This is a simple rotation; more complex rotation methods can be applied
        Fq(:, i) = U * S * V';  % Modified feature column after rotation
        
        % Calculate objective function value (e.g., using some clustering criterion)
        % Example: minimize the distance between Fq and F_rot (you can modify this)
        obj(i) = norm(F(:, i) - Fq(:, i));  % Objective function value
    end
    
    % Return rotated features and the objective function values
    F_rot = Fq;  % Final rotated feature matrix
end
