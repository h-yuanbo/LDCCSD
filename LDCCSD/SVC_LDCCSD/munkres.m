function [assignment, cost] = munkres(costMat)
% MUNKRES   Munkres (Hungarian) Algorithm for Linear Assignment Problem
% [ASSIGN,COST] = munkres(COSTMAT)
%   ASSIGN(i) = j means row i assigned to column j
%   COST is minimized cost

costMat(costMat~=costMat)=Inf; % NaN => Inf

[nRows, nCols] = size(costMat);
n = max(nRows, nCols);
if nRows ~= nCols
    costMat(n,n) = 0;
end

% Step 1: Subtract row minima
for i = 1:n
    costMat(i,:) = costMat(i,:)-min(costMat(i,:));
end

% Step 2: Subtract column minima
for j = 1:n
    costMat(:,j) = costMat(:,j)-min(costMat(:,j));
end

% Masks for stars and primes
stars = false(n);
primes = false(n);
coveredRows = false(n,1);
coveredCols = false(n,1);

% Step 3: Star zeros
for i = 1:n
    for j = 1:n
        if costMat(i,j) == 0 && ~coveredRows(i) && ~coveredCols(j)
            stars(i,j) = true;
            coveredRows(i) = true; coveredCols(j) = true;
        end
    end
end
coveredRows(:) = false;
coveredCols(:) = false;

% Step 4: Cover columns with starred zeros
for j = 1:n
    coveredCols(j) = any(stars(:,j));
end

while ~all(coveredCols)
    % Step 5: Prime uncovered zero
    zflag = true;
    while zflag
        [row,col] = find(costMat==0 & ~coveredRows & ~coveredCols',1);
        if isempty(row)
            zflag = false; break;
        end
        primes(row,col) = true;
        starCol = find(stars(row,:), 1);
        if isempty(starCol)
            % augment path
            path = [row col];
            done = false;
            while ~done
                r = find(stars(:,path(end,2)), 1);
                if isempty(r)
                    done = true; break;
                end
                path(end+1,:) = [r, path(end,2)];
                c = find(primes(path(end,1),:), 1);
                path(end+1,:) = [path(end,1), c];
            end
            for k = 1:size(path,1)
                if stars(path(k,1),path(k,2))
                    stars(path(k,1),path(k,2)) = false;
                else
                    stars(path(k,1),path(k,2)) = true;
                end
            end
            primes(:) = false;
            coveredRows(:) = false;
            coveredCols(:) = false;
            for j = 1:n
                coveredCols(j) = any(stars(:,j));
            end
            zflag = false;
        else
            coveredRows(row) = true;
            coveredCols(starCol) = false;
        end
    end
    if all(coveredCols), break; end

    % Step 6: Adjust matrix
    minval = min(costMat(~coveredRows, ~coveredCols));
    costMat(~coveredRows, ~coveredCols) = costMat(~coveredRows, ~coveredCols) - minval;
    costMat(coveredRows, coveredCols) = costMat(coveredRows, coveredCols) + minval;
end

% Form assignment
assignment = zeros(nRows,1);
for i = 1:nRows
    j = find(stars(i,1:nCols),1);
    if ~isempty(j)
        assignment(i) = j;
    else
        assignment(i) = 0;
    end
end

cost = sum(costMat((1:nRows) + (assignment'-1)*n) .* (assignment' > 0));

end
