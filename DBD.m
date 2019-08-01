function [W] = DBD(X, sw, sb, params)
% X：MDPDV数据
% K：取特征向量的个数
% Loop_num：循环次数

lambda1 = 1e-3;       % 论文数值
lambda2 = 1e-4;    % 论文数值
lambda3 = 0.01;       % 论文数值
opts.record = 0;
opts.mxitr  = 3000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
K = params.K;            % 论文数值
[n_Sample, n_Feature] = size(X);
M = mean(X,1);
data = (X - repmat(M,n_Sample,1));
[eigenvector, eigenvalue] = eig(data'*data);  % 求特征向量、特征值
[value, index]= sort(diag(eigenvalue),'descend');
W0 = eigenvector(:, index(1:K));   % 取前k个特征向量初始化W

Loop_num = 100;
W = W0;
for i = 1:Loop_num
    % 固定W，更新B
    B = double(X*W>0);
    
    % 固定B，更新W
    F1 = lambda1*(X'*X) + lambda2* sw - lambda3*sb;
    F2 = -2*lambda1*X'*B;
    [W, G]= OptStiefelGBB(W, @objectfunc,opts,F1,F2');
end
end

function [F, G] = objectfunc(W, F1, F2)
    F = trace(W'*F1*W) + trace(F2*W);
    G = 2*F1*W + F2';
end



