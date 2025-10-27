function [Cs] = MGDHSLR_updateCandA(alpha,Ct,Ks,AKs,AKt,As,P,Lambda)

    manifold.k = 10;
    manifold.Metric = 'Euclidean';
    manifold.WeightMode = 'HeatKernel';  
    manifold.NeighborMode = 'KNN'; %'Supervised';
    manifold.normr = 1;
    [Lp,~,~]=computeL(P,manifold);
    Lp=Lp./norm(Lp,'fro');

    HAs = centeringMatrix(size(As,2));
    HCs = centeringMatrix(size(As,1));

    A = alpha * AKs' * AKs;
    B = alpha * AKs' * AKt * Ct * P';
    C = 1 / 9 * Ks' * Ks;
    D = 1 / 3 * Ks' * Lambda;
    F = B + D;

    Ds = diag(sum(P, 2));
    M = A \ C;
    N = Ds;
    Q = A \ F;
    Cs = sylvester(M, N, Q);

end

function L = getL_from_BipartiteGraph(S)
    [n1,n2] = size(S);
    n = n1+n2;
    W = [zeros(n1,n1),S;S',zeros(n2,n2)];
    Dw = diag(sparse(sqrt(1 ./ (sum(W)+eps))));
    L = eye(n) - Dw * W * Dw;
end
