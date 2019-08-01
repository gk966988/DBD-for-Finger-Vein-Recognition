function fea = Extract_feature(img,W,D,params)

clusters = params.clusters;
M = params.M;
N = params.N;

Tdata = double(img);
[h, w] = size(img);

R = params.R;
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
CG = Tdata((R+1):(h-R), (R+1):(w-R));
Tcode1 = zeros(size(CG,1),size(CG,2),size(spoints,1));
for ii=1:size(spoints,1)
    Tmp = Tdata(((R+1):(h-R))+spoints(ii,1),((R+1):(w-R))+spoints(ii,2));
    Tcode1(:, :, ii) = Tmp-CG;
end

[Th, Tw, ~] = size(Tcode1);
delta_h = round(Th / M);
delta_w = round(Tw / N);

fea = zeros(1,M*N*clusters);
cnt=0;
for i = 1:M
    s_h = delta_h * (i-1) + 1;
    e_h = s_h + delta_h - 1;
    e_h = min(e_h, Th);
    for j = 1:N
        s_w = delta_w * (j-1) + 1;
        e_w = s_w + delta_w - 1;
        e_w = min(e_w, Tw);
        Tmp = Tcode1(s_h:e_h, s_w:e_w, :);
        [t_h, t_w, t_d] = size(Tmp);
        Tmp = reshape(Tmp, t_h*t_w, t_d);

        fea((cnt+1):cnt+clusters)= ...
          getHist(double(Tmp*W{i,j} > 0), D{i,j});
        cnt = cnt+clusters;
    end
end