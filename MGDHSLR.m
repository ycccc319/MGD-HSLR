function [B_trn,B_tst] = MGDHSLR(exp_data,options)
    addpath(genpath('./utils/'));
    Xs                =    exp_data.Xs';
    Xt                =    exp_data.train';
    Ys                =    exp_data.ys;
    Yt                =    exp_data.yt;
    test              =    exp_data.test;

    %% init parameters
    T=options.T;
    TR = options.TR;
    epsilon = options.epsilon;
    alpha = options.alpha;
    beta = options.beta;
    delta = options.delta;
    gamma = options.gamma;
    k = options.k;
    
    %% Init
    X=[Xs,Xt];
    X=L2Norm(X')';
    [m,ns]=size(Xs);
    nt=size(Xt,2);
    C=length(unique(Ys));
    anchorNumTotalSource = max(ceil(ns * 0.1),C);
    anchorNumTotalTarget = max(ceil(nt * 0.1),C);
    n=ns+nt;
    
    %% Init pseudo label
    Ytpseudo=classifySVM(Xs,Ys,Xt);

    %% Init Ks, Kt, As, At
    [Ks,Y_ks,Ys_pseudo,As,predY, nks, acc1, ~] = MGDHSLR_solveSelfRepresentation(Xs,Ys,2,anchorNumTotalSource,C);
    [Kt,Y_kt,Yt_pseudo,At,~, nkt, ~, ~] = MGDHSLR_solveSelfRepresentation(Xt,ones(nt,1),2,anchorNumTotalTarget,C); 

    %% Init Ds, Dt
    if gamma == 0
        Ds = zeros(nks,ns);
        Dt = zeros(nkt,nt);
    else
        Ds = 1./nks*ones(nks,ns);
        Dt = 1./nkt*ones(nkt,nt); 
    end

    %% Init Cs, Ct
    Cs = zeros(nks,ns);
    for i = 1:ns
        idx = find(Y_ks == Ys(i,1));
        Cs(idx,i) = 1./length(idx);
    end
    Ykt = classifySVM(Xs,Ys,Kt);
    Ct = zeros(nkt,nt);
    for i = 1:nt
        idx = find(Ykt == Ytpseudo(i,1));
        if ~isempty(idx)
            Ct(idx,i) = 1./length(idx);
        end
    end

    %% Init C
    Cs = 1./nks*ones(nks,ns);
    Ct = 1./nkt*ones(nkt,nt); 

    %% Init A
    As = 1./nks*ones(nks,ns);
    At = 1./nkt*ones(nkt,nt); 

    %% Init Z
    Z = eye(nt,ns);

    %% Init Es, Et
    Es = zeros(m,ns);
    Et = zeros(m,nt);

    %% Init Pc, Pa
    Pc = eye(ns,nt);
    Pa = eye(ns,nt);

    %% Init W
    W=eye(m);

    %% Print anchor's acc
    fprintf('[Init] anchor acc: %.4f\n',acc1);

    %% Calculate MMD's M0
    M0 = 0;

    %% get projection
    [AXs,AXt,AKs,AKt] = MGDHSLR_getProjection(W,X,Ks,Kt,ns,nt);

    %% DA step
    for i=1:T
        %% Update P
        Pc = zeros(ns,nt);
        for j = 1:ns
            idx = find(Ytpseudo == Ys(j,1));
            Pc(j,idx)=1/length(idx);
        end
        Pa = MGDHSLR_updatePa(min(k,nks-1),AKs,AKt,ns,nt,nks,nkt,Y_ks,Ys_pseudo,Yt_pseudo);

        %% Update D
        Ds = (Ks'*Ks+9*gamma*Ks'*W*W'*Ks)\(3*Ks'*(Xs-1/3*Ks*As-1/3*Ks*Cs-Es)+9*gamma*Ks'*W*W'*Kt*Dt*Z);
        Dt = MGDHSLR_updateDt(Ks,Kt,Ds,W,Z,(Xt-1/3*Kt*At-1/3*Kt*Ct-Et),gamma);

        %% Update Z
        AA = W' * Ks * Ds;
        BB = W' * Kt * Dt;
        [U, ~, V] = svd(BB' * AA, 'econ');
        Z = U * V';

        %% Update C
        Lambda1 = Xs-1/3*Ks*As-1/3*Ks*Ds-Es;
        Lambda2 = Xt-1/3*Kt*At-1/3*Kt*Dt-Et;
        Cs = MGDHSLR_updateCandA(alpha,Ct,Ks,AKs,AKt,As,Pc,Lambda1);
        Ct = MGDHSLR_updateCandA(alpha,Cs,Kt,AKt,AKs,At,Pc',Lambda2);

        %% Update A
        Lambda1 = Xs-1/3*Ks*Cs-1/3*Ks*Ds-Es;
        Lambda2 = Xt-1/3*Kt*Ct-1/3*Kt*Dt-Et;
        As = MGDHSLR_updateCandA(beta,At,Ks,AKs,AKt,Cs,Pa,Lambda1);
        At = MGDHSLR_updateCandA(beta,As,Kt,AKt,AKs,Ct,Pa',Lambda2);

        %% Update E
        Es = Xs  - 1/3*Ks*As - 1/3*Ks*Ds - 1/3*Ks*Cs;
        Et = Xt  - 1/3*Kt*At - 1/3*Kt*Dt - 1/3*Kt*Ct;

        manifold.k = 5; % k;
        manifold.Metric = 'Euclidean';
        manifold.WeightMode = 'Binary'; 
        manifold.NeighborMode = 'Supervised'; %'Supervised';
        manifold.normr = 1;
        Y_ktpseudo=classifySVM(AKs,Y_ks,AKt);
        manifold.gnd = [Y_ks;Y_ktpseudo];
        tabu = tabulate(manifold.gnd);
        manifold.k = min(max(k,1),min(floor(tabu(:,2)))-1);

        %% Update W
        W = MGDHSLR_updateW(Xs,Xt,...
            Ks,Kt,Cs,Ct,Pc,As,At,Pa,Ds,Dt,Z,...
            delta,alpha,beta,epsilon,gamma,manifold,...
            [AKs,AKt],[Ks,Kt]);

        %% get projection
        [AXs,AXt,AKs,AKt] = MGDHSLR_getProjection(W,X,Ks,Kt,ns,nt);

        %% 课程
        sC=2; pos=C-sC+1;
        selectRate = max(0,1 - (i/10));
        [probYt,trustable,Ytpseudo] = MGDHSLR_getDPL(AXs,Ys,AXt,Ytpseudo,pos,selectRate);
        trustable = logical(trustable);

        %% Print the accuracy
        acc_before = getAcc(Ytpseudo(trustable),Yt(trustable));
        acc_before2 = getAcc(Ytpseudo,Yt);
        trainSamp = [AXs,AXt(:,trustable)];
        trainLabel = [Ys;Ytpseudo(trustable)];
        Ytpseudo=classifySVM(trainSamp,trainLabel,AXt);
        acc = getAcc(Ytpseudo,Yt);
        Ytpseudo2=classifySVM(AXs,Ys,AXt);
        acc2 = getAcc(Ytpseudo2,Yt);
        fprintf('[%2d] acc:%.4f (%.4f => %.4f/%.4f), acc_W:%.4f  ==> ipv: %.4f \n',i,acc,...
            acc_before,acc_before2,acc-acc_before2,...
            acc2, acc-acc2);
    end
    
    %% Hash step
    [AXs,AXt,AKs,AKt] = MGDHSLR_getProjection(W,X,Ks,Kt,ns,nt);
    phi = options.phi;
    eta = options.eta;
    tau = options.tau;
    epsilon_R = options.epsilon_R;
    rng(1);
    AX = [AXs,AXt];
    R = randn(size(AX,1),options.r);
    B = sign(R'*AX);

    %% Update P
    Ytpseudo=classifySVM(AXs,Ys,AXt);
    Pc = zeros(ns,nt);
    for j = 1:ns
        idx = find(Ytpseudo == Ys(j,1));
        Pc(j,idx)=1/length(idx);
    end
    Pa = MGDHSLR_updatePa(min(k,nks-1),AXs,AXt,ns,nt,nks,nkt,Y_ks,Ys_pseudo,Yt_pseudo);

    La = getL_from_BipartiteGraph(Pa);
    Lc = getL_from_BipartiteGraph(Pc);
    
    fprintf("[Hash] ");
    for i=1:TR
        fprintf("·");
        R = (AX*AX'+(tau+epsilon_R)*eye(size(AX,1)))\(AX*B');
        B = sign((R'*AX)/(phi*Lc+eta*La+(1+epsilon_R)*eye(n)));
    end
    H = sign(B)*X'/(X*X'+epsilon_R*eye(size(X,1)));

    if strcmp(options.retrieval, 'cross-domain')
        B_train           =    (Xs'*H'>0);
    elseif strcmp(options.retrieval, 'single-domain')
        B_train           =    (Xt'*H'>0);
    end
    B_test            =    (test*H'>0);
    B_trn             =    compactbit(B_train);
    B_tst             =    compactbit(B_test);
    fprintf(" OK.\n");
end

function L = getL_from_BipartiteGraph(S)
    [n1,n2] = size(S);
    n = n1+n2;
    W = [zeros(n1,n1),S;S',zeros(n2,n2)];
    Dw = diag(sparse(sqrt(1 ./ (sum(W)+eps))));
    L = eye(n) - Dw * W * Dw;
end