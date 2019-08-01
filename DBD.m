function [W] = DBD(X, sw, sb, params)
% X��MDPDV����
% K��ȡ���������ĸ���
% Loop_num��ѭ������

lambda1 = 1e-3;       % ������ֵ
lambda2 = 1e-4;    % ������ֵ
lambda3 = 0.01;       % ������ֵ
opts.record = 0;
opts.mxitr  = 3000;
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;
K = params.K;            % ������ֵ
[n_Sample, n_Feature] = size(X);
M = mean(X,1);
data = (X - repmat(M,n_Sample,1));
[eigenvector, eigenvalue] = eig(data'*data);  % ����������������ֵ
[value, index]= sort(diag(eigenvalue),'descend');
W0 = eigenvector(:, index(1:K));   % ȡǰk������������ʼ��W

Loop_num = 100;
W = W0;
for i = 1:Loop_num
    % �̶�W������B
    B = double(X*W>0);
    
    % �̶�B������W
    F1 = lambda1*(X'*X) + lambda2* sw - lambda3*sb;
    F2 = -2*lambda1*X'*B;
    [W, G]= OptStiefelGBB(W, @objectfunc,opts,F1,F2');
end
end

function [F, G] = objectfunc(W, F1, F2)
    F = trace(W'*F1*W) + trace(F2*W);
    G = 2*F1*W + F2';
end



