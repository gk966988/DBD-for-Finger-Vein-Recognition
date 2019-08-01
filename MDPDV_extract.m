function [MDPDV, sw, sb] =MDPDV_extract(data, param)
M = param.M;
N = param.N;
train_num = param.train_num;
block_w = param.block_w;
block_h = param.block_h;
stride_h = param.stride_h;
stride_w = param.stride_w;
image_num = size(data,3);
% 暂定窗口的R为3，则线上的像素点为24个
R = param.R;
Th = size(data(:,:,1), 1); % 图片的高
Tw = size(data(:,:,1), 2); % 图片的宽
spoints = [];
for i=-R:R
    spoints=[spoints;0,i];
end

for i=-R:R
    spoints = [spoints;i,0];
end
for i=-R:R
    spoints = [spoints;i,i];
end
for i=-R:R
    spoints = [spoints;i,-i];
end
spoints(R+1,:)=[];
spoints(3*R+1,:)=[];
spoints(5*R+1,:)=[];
spoints(7*R+1,:)=[];

% spoints= [0,-8;0,-7;0,-6;0,-5;0,-4;0,-3;0,-2;0,-1;0,1;0,2;0,3;0,4;0,5;0,6;0,7;0,8;
%           -8,0;-7,0;-6,0;-5,0;-4,0;-3,0;-2,0;-1,0;1,0;2,0;3,0;4,0;5,0;6,0;7,0;8,0;
%           -8,-8;-7,-7;-6,-6;-5,-5;-4,-4;-3,-3;-2,-2;-1,-1;1,1;2,2;3,3;4,4;5,5;6,6;7,7;8,8;
%           -8,8;-7,7;-6,6;-5,5;-4,4;-3,3;-2,2;-1,1;1,-1;2,-2;3,-3;4,-4;5,-5;6,-6;7,-7;8,-8];
MDPDV = cell(M, N);  % M：行的分块数，N：列的分块数
for i=1:M
    for j=1:N
        MDPDV{i,j} = zeros( block_w * block_h, size(spoints,1), size(data,3) ); % (窗宽*窗长*图片数， 64）
    end
end
cnt = 0;

% Tdata = zeros(Th+R*2, Tw+R*2);
for d=1:size(data,3)
    [h, w] = size(data(:,:,d));
    Tdata = padarray(data(:,:,d), [R, R], 'replicate', 'both');
    Tdata( (1+R):(h+R) , (1+R):(w+R) ) = double(data(:,:,d));  % 一张图片
    %Tdata = preproc2(Tdata,0.2,1,2,[],[],10);
    
    CG = Tdata((R+1):(h+R), (R+1):(w+R));  % 一张图片
    Tcode1 = zeros(size(CG,1),size(CG,2),size(spoints,1)); % (图长，图宽，24)
    
    for ii=1:size(spoints,1)   % 1:24
        Tmp = Tdata(((R+1):(h+R))+spoints(ii,1),((R+1):(w+R))+spoints(ii,2));
        Tcode1(:, :, ii) = Tmp-CG;
    end
    
    for i = 1:M
        r = (i-1)*stride_h;
        for j = 1:N
            c = (j-1)*stride_w;
            Tmp = Tcode1(r+1:r+block_h, c+1:c+block_w, :);
%             MDPDV{i,j}(cnt+1:cnt+block_w * block_h,:) = ...
%                 reshape(Tmp, block_w * block_h, size(spoints,1));
            MDPDV{i,j}(:,:,d) = reshape(Tmp, block_w * block_h, size(spoints,1));
        end
    end
end

% 计算sw, sb
sw = cell(M,N);  % 类内间距和
sb = cell(M,N);  % 类间间距和
m = cell(M,N);   % 所有样本的平均值
% 计算所有样本的平均值
for i=1:M
    for j=1:N
        m{i,j}=mean(MDPDV{i,j}, 3);
    end
end
% 计算类间间距和

for i=1:M
    for j=1:N
        sw{i,j} = zeros(size(spoints, 1),size(spoints, 1));
        sb{i,j} = zeros(size(spoints, 1),size(spoints, 1));
        for k = 1:train_num
           sw{i,j} = sw{i,j} + (MDPDV{i,j}(:, :, k) - mean(MDPDV{i,j}(:,:, 1:train_num), 3))'*(MDPDV{i,j}(:, :, k) - mean(MDPDV{i,j}(:,:, 1:train_num), 3));
        end
        sb{i,j} =sb{i,j} + 6*(mean(MDPDV{i,j}(:,:,1:train_num), 3)-m{i,j})'*(mean(MDPDV{i,j}(:,:,1:train_num),3)-m{i,j});
    end
end

for i=1:M
    for j=1:N
        for k = 2*train_num:train_num:image_num
            for z = k-train_num+1:k
                sw{i,j} = sw{i,j} + (MDPDV{i,j}(:, :, z) - mean(MDPDV{i,j}(:,:, k-train_num+1:k), 3))'*(MDPDV{i,j}(:, :, z) - mean(MDPDV{i,j}(:,:, k-train_num+1:k), 3));
            end
%             sw{i, j} = sw{i, j} + (bsxfun(@minus,MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :),mean(MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :))))'* (bsxfun(@minus,MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :),mean(MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :))));
%             sw{i, j} = sw{i,j} + var(MDPDV{i, j}(block_w*block_h*(k-2):block_w*block_h*k, :), 0, 1);
%             sb{i,j} =sb{i,j} + 3*(mean(MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :))-m{i,j}).^2;
%             sb{i,j} = 3*(mean(MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :))-m{i,j})'*(mean(MDPDV{i,j}(block_w*block_h*(k-2):block_w*block_h*k, :))-m{i,j});
            sb{i,j} = sb{i,j} + train_num*(mean(MDPDV{i,j}(:,:,k-train_num+1:k), 3)-m{i,j})'*(mean(MDPDV{i,j}(:,:,k-train_num+1:k),3)-m{i,j});
        end
    end
end

for i=1:M
    for j=1:N
        MDPDV{i,j} = permute(MDPDV{i,j},[2, 1, 3]);
        MDPDV{i,j} = reshape(MDPDV{i,j},size(spoints,1),[])';
    end
end






