function [Dt] = MGDHSLR_updateDt(Ks,Kt,Ds,W,Z,Lambda,gamma)
    A = Kt'*Kt;
    B = 3*Kt'*Lambda;
    C = 9*gamma*Kt'*W*W'*Kt;
    D = 9*gamma*Kt'*W*W'*Ks*Ds*Z';
    
    M = C \ A;
    N = Z*Z';
    Q = C \ (B+D);
    Dt = sylvester(M, N, Q);
    
end

