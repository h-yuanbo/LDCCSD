function [nbrIdx] = build_knn_graph(X, k)
    % Compute k-NN (excluding self). Returns:
    %  - nbrIdx: n x k integer indices, each row i lists neighbors N_i
    %  - O_of_i{i}: the set O_i = { o | i in N_o }
    %  - pos_in_O{i}: same length as O_of_i{i}, position v in N_o where i sits
    n = size(X,1);
    D = pdist2(X,X,'euclidean');
    D(1:n+1:end) = inf;                 % exclude self
    [~, order] = sort(D, 2, 'ascend');
    nbrIdx = order(:,1:k);              % k nearest neighbors per row

    O_of_i = cell(n,1);
    pos_in_O = cell(n,1);
    for o = 1:n
        for v = 1:k
            i = nbrIdx(o,v);
            O_of_i{i}  = [O_of_i{i}, o];
            pos_in_O{i}= [pos_in_O{i}, v];
        end
    end
end

