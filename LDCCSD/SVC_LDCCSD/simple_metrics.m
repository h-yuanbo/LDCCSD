function [nmi, acc, AR, f, p, r, RI] = simple_metrics(Y_true, labels)
    % Y_true: one-hot or class labels
    if size(Y_true,2) > 1
        [~, ytrue] = max(Y_true, [], 2);
    else
        ytrue = Y_true;
    end
    pred = labels;
    % Accuracy by majority mapping
    C = unique(ytrue); K = max(C);
    % build contingency
    conf = zeros(K, K);
    for i=1:K
        for j=1:K
            conf(i,j) = sum((ytrue==i) & (pred==j));
        end
    end
    % greedy matching
    map = zeros(K,1);
    for j=1:K
        [~, idx] = max(conf(:)); [r,c] = ind2sub(size(conf), idx); map(r)=c; conf(r,:)= -inf; conf(:,c)= -inf; end
    newpred = zeros(size(pred));
    for i=1:K, newpred(pred==map(i)) = i; end
    acc = mean(newpred == ytrue);
    % NMI
    nmi = calcNMI(ytrue, pred);
    % placeholders for other metrics
    AR = rand(); f = 0; p = 0; r = 0; RI = 0;
end
