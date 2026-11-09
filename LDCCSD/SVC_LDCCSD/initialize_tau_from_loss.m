function tau_initial = initialize_tau_from_loss(trace_vec, initial_selection_percentage)
% INITIALIZE_TAU_FROM_LOSS calculates the initial tau (tau_0) for self-paced learning.
% This function is called within the main algorithm loop, specifically on the first iteration (it == 1).
% It uses the 'trace_vec' (individual sample losses) that have just been computed
% after the first Y update, ensuring that tau_0 is relevant to the current loss landscape.
%
%   Inputs:
%       trace_vec:           A vector (n x 1) containing the individual loss values L_i(Y)
%                            for each sample, computed based on the current Y and X.
%       initial_selection_percentage: The percentage (as a decimal, e.g., 0.20 for 20%)
%                                     of samples with the lowest loss to be initially
%                                     considered "easy" (i.e., u_i = 1).
%
%   Output:
%       tau_initial: The calculated initial value for the self-paced parameter tau.

    n = length(trace_vec); % Get the total number of samples from the trace_vec length

    % Input validation for percentage_selected
    if initial_selection_percentage <= 0 || initial_selection_percentage > 1
        error('initial_selection_percentage must be between 0 and 1 (exclusive of 0).');
    end

    % 1. Sort the individual sample loss values in ascending order.
    sorted_losses = sort(trace_vec, 'ascend');

    % 2. Determine the number of samples that should be initially selected.
    % Use 'max(1, ...)' to ensure at least one sample is selected, even if 'n * percentage' is very small.
    num_selected_samples = max(1, floor(n * initial_selection_percentage));

    % 3. The initial tau (tau_0) is set to the loss value of the
    %    'num_selected_samples'-th sample in the sorted list. This threshold
    %    ensures that exactly 'num_selected_samples' samples will have u_i = 1 (or more,
    %    if there are ties in loss values exactly at tau_0).
    tau_initial = sorted_losses(num_selected_samples);

    % Optional: You might want to print this value for debugging or monitoring
    % fprintf('  INFO: Initial tau (tau_0) set to %.6f to select approximately %d (%.2f%%) samples.\n', ...
    %         tau_initial, num_selected_samples, initial_selection_percentage * 100);

end
