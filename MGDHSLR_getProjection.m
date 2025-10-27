function [AXs,AXt,AKs,AKt] = MGDHSLR_getProjection(W,X,Ks,Kt,ns,nt)
    nks = size(Ks,2);
    nkt = size(Kt,2);
    n = ns+nt;
    AX=W'*[X,Ks,Kt];
    AX=L2Norm(AX')';
    AXs=AX(:,1:ns);
    AXt=AX(:,ns+1:ns+nt);
    AKs=AX(:,n+1:n+nks);
    AKt=AX(:,n+nks+1:n+nks+nkt);
end

