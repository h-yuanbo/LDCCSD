function [res]= my_eval_y_mismatch(predY,Y)
%  This code is used for the case when the cluster number in the predict
%  label is not the number of true cluster
%
if size(Y,2) ~= 1
    Y = Y';
end
if size(predY,2) ~= 1
    predY = predY';
end

nSmp = length(Y);
uY = unique(Y);
nclass = length(uY);
Y0 = zeros(nSmp,1);
if nclass ~= max(Y)
    for i = 1:nclass
        Y0(Y == uY(i)) = i;
    end
    Y = Y0;
end

uY = unique(predY);
nclass = length(uY);
predY0 = zeros(nSmp,1);
if nclass ~= max(predY)
    for i = 1:nclass
        predY0(predY == uY(i)) = i;
    end
    predY = predY0;
end


[newIndx] = bestMap_v2(Y,predY);
acc = mean(Y==newIndx);
nmi = mutual_info(Y,newIndx);
purity = pur_fun(Y,newIndx);
[AR,RI,MI,HI] = RandIndex(Y, newIndx);
[fscore,precision,recall] = compute_f(Y, newIndx);

nCluster = length(unique(Y));
nSmp = length(Y);
FF = zeros(nSmp, nCluster);
for iSmp = 1 : nSmp
    FF(iSmp, predY(iSmp))=1;
end
ys = sum(FF);
[entropy,bal, SDCS, RME] = BalanceEvl(nCluster, ys);
res = [acc; nmi; purity; AR; RI; MI; HI; fscore; precision; recall; entropy; SDCS; RME; bal;];