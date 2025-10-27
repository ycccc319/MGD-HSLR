function [W] = MGDHSLR_updateW(Xs,Xt,...
    Ks,Kt,Cs,Ct,Pc,As,At,Pa,Ds,Dt,Z,...
    delta,alpha,beta,epsilon,gamma,manifold,...
    Lpp_X,Lpp_in_X)
    X = [Xs,Xt];
    K = [Ks,Kt];
    [Xm,Xn] = size(X);
    [Km,Kn] = size(K);

    %% Part B: calculate K's Laplacian regularization
    [Lk,~,~]=computeL(Lpp_X,manifold);
    Lk=Lk./norm(Lk,'fro');
    BB = delta*Lpp_in_X*Lk*Lpp_in_X';

    %% Part C: calculate all about C
    Dcs = diag(sum(Pc, 2));  % ns x ns
    Dct = diag(sum(Pc, 1));  % nt x nt
    % 2817*2817 298*298 2817*298
    CC = alpha*(Ks*Cs*Dcs*Cs'*Ks'+Kt*Ct*Dct*Ct'*Kt'-2*Ks*Cs*Pc*Ct'*Kt');

    %% Part D: calculate all about A
    Das = diag(sum(Pa, 2));  % ns x ns
    Dat = diag(sum(Pa, 1));  % nt x nt
    % 2817*2817 298*298 2817*298
    DD = beta*(Ks*As*Das*As'*Ks'+Kt*At*Dat*At'*Kt'-2*Ks*As*Pa*At'*Kt');

    %% Part E: calculate all about D
    EE = gamma*(Ks*Ds*Ds'*Ks'+Kt*Dt*Z*Z'*Dt'*Kt' ...
        -Ks*Ds*Z'*Dt'*Kt'-Kt*Dt*Z*Ds'*Ks');

    %% Final: calculate W
    I = eye(Xm,Xm);
    W = inv(BB+CC+DD+EE+epsilon*I);

end