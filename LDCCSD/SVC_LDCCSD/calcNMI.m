function nmi = calcNMI(true_labels, pred_labels)
    contingency = crosstab(true_labels, pred_labels);
    N = sum(contingency(:));
    Pi = sum(contingency,2); Pj = sum(contingency,1);
    % mutual information
    nz = contingency>0;
    MI = sum(contingency(nz) ./ N .* log((contingency(nz).*N)./(Pi(nz(:,1)).*Pj(nz(1,:)) + eps) + eps));
    H_true = - sum((Pi./N) .* log(Pi./N + eps));
    H_pred = - sum((Pj./N) .* log(Pj./N + eps));
    nmi = MI / ((H_true + H_pred)/2 + eps);
end
