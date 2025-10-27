function [K,labelOfAnchor,Ypseudo,Bs,predY, nk, acc, sse] = MGDHSLR_solveSelfRepresentation(X,Y,anchorCalWay,anchorNumEachClass, realC)
    if anchorCalWay == 3
        [best_k, SSE_values, k_values] = elbow_method(X, size(X,2));
        anchorNumEachClass = best_k;
        disp(best_k);
    end
    C = length(unique(Y));
    K = [];
    labelOfAnchor = [];
    total = 0;
    for i = 1:C
        ind = (Y == i);
        Xsc =  X(:,ind);
        nsc = size(Xsc,2);
        if anchorCalWay == 0
            if anchorNumEachClass >= nsc
               anchorNum = max(1, ceil(nsc/2)); 
            else
               anchorNum = anchorNumEachClass;
            end
        elseif anchorCalWay == 1
            [best_k, SSE_values, k_values] = elbow_method(Xsc, min(C,nsc));
            if C > 1
                anchorNum = max(ceil(nsc*0.1),best_k);
            else
                anchorNum = max(max(realC*2,ceil(nsc*0.1)),best_k);
            end
            total = total + anchorNum;
        elseif anchorCalWay == 4
            clusterer = HDBSCAN( Xsc' );
            clusterer.minpts = 2;
            clusterer.minclustsize = 2;
            clusterer.outlierThresh = 0.9;
            clusterer.minClustNum = ceil(20);
            clusterer.fit_model(1,false); 
            clusterer.get_best_clusters();
            clusterer.get_membership();
            test_labels = clusterer.labels;
            Xtest = Xsc(:,1)';
            [newLabels,probability] = clusterer.predict( Xsc(:,1)' );
            anchorNum = max(ceil(nsc*0.1),max(test_labels));
            disp(anchorNum);
            total = total + anchorNum;
        else
            anchorNum = ceil(anchorNumEachClass*size(Xsc,2)/size(X,2));
            total = total + anchorNum;
        end
        % initialize the anchors by VDA so that the performance is stable
        % 使用确定的方法选择锚点，以确保性能的稳定性
        [ CX, ~, ~ ] = MGDHSLR_VDA(Xsc', anchorNum); CX = CX'; 
        [Ksc,sse2]=vgg_kmeans(Xsc,anchorNum, CX);
        K = [K, Ksc];
        labelOfAnchor = [labelOfAnchor; ones(size(Ksc,2),1)*i];
    end
    % orth 奇异值分解，锚点正交化
    [U,~,V] = svd(K,'econ');
    K = U*V';
    %
    nk = size(K,2);
    dist = EuDist2(X',K'); % 每个样本到每个锚点的距离
    % 依据最近的锚点得出每个样本属于哪个锚点
    [val,Ypseudo] = min(dist,[],2); % Ypseudo = classifyKNN((K')',1:nk,(X')',1);
    Bs = hotmatrix(Ypseudo,nk,0)';
    [~,predY] = max(Bs'*hotmatrix(labelOfAnchor,C,0),[],2); % 计算所属类别
	acc = getAcc(predY,Y); % 得到锚点精准度
    sse = sum(val);
end


function [bestK, Gap, sk] = gapStatistic(X, C, B)
    % GAPSTATISTIC 使用Gap Statistic确定最佳聚类簇数量
    %   输入:
    %       X - d*n的数据集矩阵，d为维度，n为样本数
    %       C - 给定的类别数量，用于计算K的上限为floor(n/C)
    %       B - 可选，参考数据集的数量，默认10
    %   输出:
    %       bestK - 确定的最佳簇数量
    %       Gap - 每个K对应的Gap值
    %       sk - 每个K对应的标准误差
    
    % 处理默认参数
    if nargin < 3
        B = 10;  % 默认生成10个参考数据集
    end
    
    % 基本参数计算
    n = size(X, 2);  % 样本数量
    K_max = n;%floor(n / C);
    K_range = 2:K_max;
    
    % 输入合法性检查
    if length(K_range) == 0
        error('K的范围为空，请检查C的值是否过小或样本数量是否足够');
    end
    
    % 初始化存储变量
    Gap = zeros(length(K_range), 1);
    sk = zeros(length(K_range), 1);
    
    % 循环计算每个K值的Gap统计量
    for i = 1:length(K_range)
        K = K_range(i);
        
        % 1. 计算实际数据的WCSS
        [~, ~, sumd] = kmeans(X', K, 'Replicates', 10);
        Wk = sum(sumd);  % 总平方误差和
        logWk = log(Wk);
        
        % 2. 生成B个参考数据集并计算WCSS
        logWkb = zeros(B, 1);
        for b = 1:B
            % 生成均匀分布的参考数据集
            X_ref = zeros(size(X));
            for j = 1:size(X, 1)  % 每个维度独立生成
                x_min = min(X(j, :));
                x_max = max(X(j, :));
                X_ref(j, :) = x_min + (x_max - x_min) * rand(1, n);
            end
            
            % 计算参考数据集的WCSS
            [~, ~, sumd_ref] = kmeans(X_ref', K, 'Replicates', 10);
            Wkb = sum(sumd_ref);
            logWkb(b) = log(Wkb);
        end
        
        % 3. 计算Gap值和标准误差
        Gap(i) = mean(logWkb) - logWk;
        sk(i) = std(logWkb) * sqrt(1 + 1/B);  % 标准误差计算
    end
    
    % 确定最佳K值（基于Gap统计量准则）
    bestK = K_range(end);  % 默认取最大K
    for i = 1:length(K_range)-1
        if Gap(i) >= Gap(i+1) - sk(i+1)
            bestK = K_range(i);
            break;  % 找到第一个满足条件的K
        end
    end

end

function [best_k, SSE_values, k_values] = elbow_method(X, C)
    % 肘部法确定最佳聚类簇数
    % 输入：
    %   X: d*n 数据矩阵（每列一个样本）
    %   C: 类别数量（用于计算最大簇数k_max=floor(n/C)）
    % 输出：
    %   best_k: 推荐最佳簇数
    %   SSE_values: 各簇数对应的SSE值
    %   k_values: 簇数取值范围
    
    % 输入验证
    if ~ismatrix(X)
        error('输入X必须为矩阵格式');
    end
    if ~isscalar(C) || C <= 0 || mod(C, 1) ~= 0
        error('类别数量C必须为正整数');
    end

    PCA_dim = 32;
    % 1. 数据中心化（关键步骤：PCA必须基于中心化数据）
    mu = mean(X, 1);  % 计算每个特征的均值（按列求均值）
    X_centered = X - mu;  % 中心化：每个样本减去对应特征的均值
    
    % 2. 用SVD计算主成分（数值稳定性优于eig，推荐用于PCA）
    [~, ~, V] = svd(X_centered, 'econ');  % V的列向量是主成分（已按奇异值降序排列）
    
    % 3. 确保PCA维度不超过实际可用特征数
    PCA_dim = min(PCA_dim, size(V, 2));  % 防止维度超出范围
    PCA = V(:, 1:PCA_dim);  % 取前PCA_dim个主成分
    
    % 4. 对Xs和Xt进行中心化并投影（使用合并数据的均值，保证空间一致性）
    X_proj = (X - mu) * PCA;  % Xs降维结果（n_samples_xs × PCA_dim）
    
    % 更新原变量（可选）
    X = X_proj;
    
    n = size(X, 2);  % 样本数量
    k_min = 2;
%     k_max = floor(n / C);
    k_max = floor(n);
    
    % 检查簇数范围有效性
    if k_max < k_min
%         error(['计算得到的最大簇数k_max=', num2str(k_max), '小于最小簇数k_min=2，请检查C值']);
        k_max = k_min;
    end
    
    k_values = k_min:k_max;
    SSE_values = zeros(1, length(k_values));
    
    % 遍历各簇数计算SSE
    for i = 1:length(k_values)
        k = k_values(i);
        % 使用k-means聚类（重复5次避免局部最优）
        [~, ~, SSE] = kmeans(X', k, 'Replicates', 5, 'Display', 'off');
        SSE_values(i) = sum(SSE);
    end
    
    % 绘制肘部图
%     figure('Position', [100, 100, 800, 500]);
%     plot(k_values, SSE_values, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 8);
%     xlabel('簇数量 k', 'FontSize', 12);
%     ylabel('平方误差和 SSE', 'FontSize', 12);
%     title('肘部法确定最佳簇数', 'FontSize', 14);
%     grid on;
%     set(gca, 'FontSize', 10);
    
    % 自动检测肘部点（基于点到直线距离最大化）
    x = k_values;
    y = SSE_values;
    [A, B, C_line] = line_coefficients(x(1), y(1), x(end), y(end));
    distance = abs(A*x + B*y + C_line) ./ sqrt(A^2 + B^2);
    [~, idx] = max(distance);
    best_k = x(idx);
    
    % 标记肘部点
%     hold on;
%     plot(best_k, SSE_values(idx), 'rs', 'MarkerSize', 10, 'LineWidth', 2);
%     text(best_k+0.1, SSE_values(idx), ['推荐k=', num2str(best_k)], ...
%          'FontSize', 11, 'VerticalAlignment', 'bottom');
%     hold off;
end

function [A, B, C] = line_coefficients(x1, y1, x2, y2)
    % 计算直线Ax + By + C = 0的系数
    A = y2 - y1;
    B = x1 - x2;
    C = x2*y1 - x1*y2;
end