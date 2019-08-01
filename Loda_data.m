clear all;
clc;

register_path='.\data\Register\';
test_path='.\data\TestImage\';
register_image_num = 360;   % ע��ͼƬ����
test_image_num = 525;   % ����ͼƬ����
width = 96;
height = 64;


data = zeros( height, width, register_image_num);
label = cell(1, register_image_num);
register_fingerID = dir(register_path);   % ��ȡ�ļ���
N=size(register_fingerID,1);  
% CLASSES = N - 2;
num = 0;
for i = 3:N
    fingerdir=strcat(register_path,register_fingerID(i,1).name,'\');
    redister_IDname = register_fingerID(i, 1).name;
    file=dir(fingerdir);    % ��ȡͼƬ
    N1=size(file,1);  
    
    pre_class = N1 - 2;  % ÿ��ID ͼƬ�ĸ���
    for j=3:N1
        filename=strcat(fingerdir,file(j,1).name);
        img= imread(filename);
        
        img = rgb2gray(img);
%         figure(1);
%         imshow(img);
%         img = imresize(img,[height, width], 'bicubic');
        img = double(img);
%         figure(2);
%         imshow(img);
        data(:, :, num+j-2) = img;
        label{1, num+j-2} = redister_IDname;
    end   
    num = num + N1 - 2;
end
save train data label


data = zeros( height, width, test_image_num);
label = cell(1, test_image_num);
test_fingerID = dir(test_path);   % ��ȡ�ļ���
N=size(test_fingerID,1);  
% CLASSES = N - 2;

num_2 = 0;
for i = 3:N
    fingerdir=strcat(test_path,test_fingerID(i,1).name,'\');
    test_IDname = test_fingerID(i, 1).name;
    file=dir(fingerdir);    % ��ȡͼƬ
    N1=size(file,1);  
    
    pre_class = N1 - 2;  % ÿ��ID ͼƬ�ĸ���
    for j=3:N1
        filename=strcat(fingerdir,file(j,1).name);
        img= imread(filename);
        
        img = rgb2gray(img);
        img = double(img);
        data(:, :, num_2+j-2) = img;
        label{1, num_2+j-2} = test_IDname;
    end   
    num_2 = num_2 + N1 - 2;
end
save test data label






