%% Title: Multi-Granularity Decomposition and Hierarchical Semantics Learning for Domain Adaptation Retrieval
clc; clear all;
addpath(genpath('./utils/'));
db_name   =   ["Office31-amazon&webcam"];

options=defaultOptions('T',10,... % DA's iter time
                'TR',10,...% hash's iter time
                'alpha',0.1,... % class level
                'beta',0.01,.... % anchor level
                'gamma',0.1,... % domain level
                'delta',1,... % K's Laplace
                'phi',1,... % hash's class level
                'eta',0.01,... % hash's anchor level
                'tau',0.01,... % hash R's regular term
                'k',3,... % anchor's KNN
                'epsilon',1,... % DA's regular term
                'epsilon_R',0.1); % hash's  regular term
param.ReducedDim = 512; % The dimension reduced before training
param.retrieval = "cross-domain";
options.retrieval = "cross-domain";
options.r = 64;
runtimes = 10;

for db = 1:length(db_name)
    param.choice= 'evaluation_PR_MAP'; %  evaluation_PR
    param.retrieval= 'cross-domain'; %'single-domain' or  'cross-domain'
    param.pos = [1:10:40 50:20:400];
    now_db_name = db_name(db);

    %% run
    mean_mAP = [];
    for k = 1:runtimes
        rng('shuffle');
        fprintf('Run time: %.d\n', k);
        [exp_data] = construct_dataset(now_db_name,param);
        
        [B_trn,B_tst] = MGDHSLR(exp_data,options);

        Dhamm = hammingDist(B_tst, B_trn);
        [~, rank] = sort(Dhamm, 2, 'ascend');
        clear B_tst B_trn;
        [recall, precision, ~] = recall_precision(exp_data.WTT, Dhamm);
        [mAP] = area_RP(recall, precision);
        fprintf('mAP: %.4f\n', mAP);
        mean_mAP = [mean_mAP, mAP];
        clear exp_data;
    end
end
fprintf('Mean mAP: %.4f\n',mean(mean_mAP));
