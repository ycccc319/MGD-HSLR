function [Pa] = MGDHSLR_updatePa(k,AKs,AKt,ns,nt,nks,nkt,Y_ks,Ys,Yt)
    
    if k == 0
        Pa = zeros(ns,nt);
        return;
    end

    [idx,d] = knnsearch(AKs',AKt','k',k);
    expMatrixS = zeros(nks,nkt);
    expd = exp(-d);

    for i = 1:nkt
        expd(i,:) = expd(i,:) ./ expd(i,1);
    end

    for i = 1:nkt
        expMatrixS(idx(i,:),i) = expd(i,:);
    end

    if nargin == 10
        Pa = zeros(ns,nt);
    
        for i = 1:nks
            for j = 1:nkt
                Pa(Ys == i, Yt == j) = expMatrixS(i,j);
            end
        end
    
        Pa = L2Norm(Pa);
    else
        Pa = L2Norm(expMatrixS);
    end


end

