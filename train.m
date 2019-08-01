clear all;
clc;
params.ht = 64;    % 图像宽
params.wt = 96;    % 图像高
params.train_num = 5;     % 训练集每类图片的数量
params.block_w = 24;      % block宽
params.block_h = 64;      % block高
params.stride_w = 24;     % block水平方向滑动步长
params.stride_h = 64;     % block垂直方向滑动步长
params.R = 3;
params.N = floor((params.wt - params.block_w)/params.stride_w)+1;   % 水平方向block数
params.M = floor((params.ht - params.block_h)/params.stride_h)+1;   % 垂直方向block数
params.lambda1 = 0.001;   
params.lambda2 = 0.0001;
params.binsize = 15;
params.clusters = 300;        % kmeans聚类中心数
params.n_iter = 15;           % 最大迭代次数
params.K = 15;
load train.mat
train_label = label;
[MDPDV, sw, sb] = MDPDV_extract(data, params);
clusters = params.clusters;   % 聚类中心
M = params.M;
N = params.N;

options = zeros(1,14);
options(1) = 1; % display
options(2) = 1;
options(3) = 0.001; % precision  0.1
options(5) = 1; %centers initialize
centers = zeros(params.clusters, params.K );
options(14) = 200; % maximum iterations



for i = 1:M
    i
    for j = 1:N
        matrix = zeros(clusters, 15);
        W{i,j} = DBD(MDPDV{i,j}, sw{i,j}, sb{i,j}, params);
%         [l, c] = kmeans(double(MDPDV{i,j}*W{i,j} >0), clusters,'rep', 10);
%         D{i,j} = c; 
        [row,col] = size(MDPDV{i,j}*W{i,j});
        centers = zeros(params.clusters, params.K );
        for k=100000:100000:row
            a = MDPDV{i,j}(k-99999:k,:)*W{i,j};
            D{i,j} = sp_kmeans(centers, (double(a >0)), options);
            centers = D{i,j};
        end
    end  
end
Num = size(data, 3);
Dim = 300;
train_feature = zeros(Num, M*N*clusters);
for i = 1:Num
    image = double(data(:,:,i));
    train_feature(i,:)=Extract_feature(image, W, D, params);
end
[eigvec2,eigval,~,sampleMean] = PCA(train_feature);  % PCA降维
eigvec = (bsxfun(@rdivide,eigvec2',sqrt(eigval))');
xf = bsxfun(@minus, train_feature, mean(train_feature))*eigvec(:,1:Dim);
% save W;
% save D;
% Dim = 500;
load test.mat
% load W;
% load D;
% load sampleMean;
test_label=label;
Num_test = size(data, 3);
test_feature = zeros(Num_test, M*N*clusters);
for i = 1:Num_test
    image = double(data(:,:,i));
    test_feature(i,:)=Extract_feature(image, W, D, params);
end
eigvec_t = bsxfun(@minus,test_feature,sampleMean)*eigvec(:,1:Dim);
count = 1;
count1 = 1;
count2 = 1;
self_score = [];
other_score = [];
num_class = 72;   % 注册数据的类别数
num_correct = 0;
for i = 1:Num_test  
    i
    sim = pdist2(eigvec_t(i,:),xf,'cosine');
    [value, index] = sort(sim);
    [score_temp, location] = min(sim);
    % 计算acc用的数据
    if strcmp(test_label{1, i}, train_label{1,location})==1
        num_correct = num_correct + 1;
    end
    
    % 计算EER用的数据
    for j = 1:num_class
        [score_temp, location]=min(sim((j-1)*5+1:j*5));
        if strcmp(test_label{1, i}, train_label{1,j*5})==1
            self_score(count1,1)= score_temp;
            count1=count1+1;
        end
        if strcmp(test_label{1, i}, train_label{1,j*5})==0
            other_score(count2,1)= score_temp;
            count2=count2+1;
        end
    end
end
acc = num_correct / Num_test
other_sort=sort(other_score,'ascend');
self_sort=sort(self_score,'descend');
N=30000;
tp=zeros(N,1);
fp=zeros(N,1);
roc=0.0;

for i = 1:N
    threshold(i,1)=other_sort(ceil(i*size(other_sort ,1)/N),1);
    R_num=size(find(self_score<threshold(i,1)),1);
    W_num=size(find(other_score<threshold(i,1)),1);
    tp(i,1)=R_num/size(self_score,1);
    fp(i,1)=W_num/size(other_score,1);
    if(i>1)
    dfp = fp(i,1) - fp(i-1,1);
	mtp = (tp(i,1) + tp(i-1,1))/2;
	roc = roc + mtp * dfp;
    end
end
roc
RR=abs(tp-(1-fp));
tt=find(RR==min(RR));
EER=1-(tp(tt(1,1),1)+1-fp(tt(1,1),1))/2;
EER=EER*100
load chirp
sound(y,Fs)




   
