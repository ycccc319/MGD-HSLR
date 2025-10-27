function [ anchor, ind2, score ] = MGDHSLR_VDA(X, m)

[n,~] = size(X);
X_std = std(X,[],2);

score = X_std.^2;
score(:,1) = score/max(score);
[~,ind(1)] = max(score);
for i=2:m
    for j=1:n
        % 计算当前样本j与上一个锚点(ind(i-1))的相似度
        A_1 = score(ind(i-1),i-1);
        A_2 = score(j,i-1);
        Co(j,:)=(1 + norm(A_1-A_2,2)^(0.5))^(-1);
    end
    pho = Co/max(Co);
    % 更新得分：降低与已选锚点相似样本的权重
    score(:,i) = score(:,i-1).*(ones(n,1)-pho);
    score(:,i) = score(:,i)/max(score(:,i));
    [~,ind(i)] = max(score(:,i));
end
ind2 = sort(ind,'ascend');
anchor = X(ind2,:);
end