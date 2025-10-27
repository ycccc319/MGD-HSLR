function [exp_data]=construct_dataset(db_name,param)

addpath('./dataset/');

%% choose and load dataset
%% Office31
if strcmp(db_name,'Office31-amazon&webcam')
    data = readmatrix(['./dataset/amazon_amazon.csv']);
    Xs = data(:,1:2048);
    Xs = normalize1(Xs);
    ys = data(:,2049);
    ys = ys + 1;
    data = readmatrix(['./dataset/amazon_webcam.csv']);
    Xt = data(:,1:2048);
    Xt = normalize1(Xt);
    yt = data(:,2049);
    yt = yt + 1;
    clear data;
elseif strcmp(db_name,'Office31-webcam&amazon')
    data = readmatrix(['./dataset/webcam_webcam.csv']);
    Xs = data(:,1:2048);
    Xs = normalize1(Xs);
    ys = data(:,2049);
    ys = ys + 1;
    data = readmatrix(['./dataset/webcam_amazon.csv']);
    Xt = data(:,1:2048);
    Xt = normalize1(Xt);
    yt = data(:,2049);
    yt = yt + 1;
    clear data;
elseif strcmp(db_name,'Office31-amazon&dslr')
    data = readmatrix(['./dataset/amazon_amazon.csv']);
    Xs = data(:,1:2048);
    Xs = normalize1(Xs);
    ys = data(:,2049);
    ys = ys + 1;
    data = readmatrix(['./dataset/amazon_dslr.csv']);
    Xt = data(:,1:2048);
    Xt = normalize1(Xt);
    yt = data(:,2049);
    yt = yt + 1;
    clear data;
elseif strcmp(db_name,'Office31-dslr&amazon')
    data = readmatrix(['./dataset/dslr_dslr.csv']);
    Xs = data(:,1:2048);
    Xs = normalize1(Xs);
    ys = data(:,2049);
    ys = ys + 1;
    data = readmatrix(['./dataset/dslr_amazon.csv']);
    Xt = data(:,1:2048);
    Xt = normalize1(Xt);
    yt = data(:,2049);
    yt = yt + 1;
    clear data;
elseif strcmp(db_name,'Office31-webcam&dslr')
    data = readmatrix(['./dataset/webcam_webcam.csv']);
    Xs = data(:,1:2048);
    Xs = normalize1(Xs);
    ys = data(:,2049);
    ys = ys + 1;
    data = readmatrix(['./dataset/webcam_dslr.csv']);
    Xt = data(:,1:2048);
    Xt = normalize1(Xt);
    yt = data(:,2049);
    yt = yt + 1;
    clear data;
elseif strcmp(db_name,'Office31-dslr&webcam')
    data = readmatrix(['./dataset/dslr_dslr.csv']);
    Xs = data(:,1:2048);
    Xs = normalize1(Xs);
    ys = data(:,2049);
    ys = ys + 1;
    data = readmatrix(['./dataset/dslr_webcam.csv']);
    Xt = data(:,1:2048);
    Xt = normalize1(Xt);
    yt = data(:,2049);
    yt = yt + 1;
    clear data;
end
%% Run PCA to reduce the dimensionality 
PCA_dim = param.ReducedDim;
X = [Xs; Xt];

mu = mean(X, 1);
X_centered = X - mu;
[~, ~, V] = svd(X_centered, 'econ');
PCA_dim = min(PCA_dim, size(V, 2));
PCA = V(:, 1:PCA_dim);
Xs_proj = (Xs - mu) * PCA;
Xt_proj = (Xt - mu) * PCA;

Xs = Xs_proj;
Xt = Xt_proj;

%% Processing data
[ndatat,~]      =     size(Xt);
R = randperm(ndatat);
exp_data.R = R;
num_test = 200;
exp_data.num_test = num_test;
test            =     Xt(R(1:num_test),:);
ytest           =     yt(R(1:num_test));
R(1:num_test)   =     [];
train           =     Xt(R,:);
train_ID = R;
Yt  =yt;
yt              =     yt(R);

Y=sparse(1:length(ys), double(ys), 1);
Y=full(Y);

mdl=fitcknn(Xs,ys);
ytnew=predict(mdl,train);
num_train       =     size(train,1);
DtrueTestTrain  =    distMat(test,train);
[~,idx]         =    sort(DtrueTestTrain,2);
WtrueTestTrain  =    zeros(num_test,num_train);
for i=1:num_test
    WtrueTestTrain(i,idx(i,:)) =1;
end

if strcmp(param.retrieval, 'cross-domain')
    YS            =  repmat(ys,1,length(ytest));
    YT            =  repmat(ytest,1,length(ys));
elseif strcmp(param.retrieval, 'single-domain')
    YS            =  repmat(yt,1,length(ytest));
    YT            =  repmat(ytest,1,length(yt));
end
WTT           =  (YT==YS');

X=[Xs;Xt];
samplemean              = mean(X,1);
X = (double(X)-repmat(samplemean,size(X,1),1));
Xs                      = Xs-repmat(samplemean,size(Xs,1),1);
train                   = train-repmat(samplemean,size(train,1),1);
test                    = test-repmat(samplemean,size(test,1),1);

exp_data.db_data=X;
exp_data.num_test   = num_test;
exp_data.Xs         =   Xs ;
exp_data.train_ID   = train_ID;
exp_data.test       =   test;
exp_data.train      =   train;
exp_data.ys         =   ys ;
exp_data.yt         =   yt ;
exp_data.Yt         =   Yt ;
exp_data.train_all  =   [Xs;train];
exp_data.WTT        =   WTT ;
exp_data.Y=Y;
exp_data.ytnew      =   ytnew ;
exp_data.WtrueTestTraining = WtrueTestTrain;
exp_data.db_name = db_name;
% exp_data.WtrueTestTrainingall=WtrueTestTrainingall;


end
