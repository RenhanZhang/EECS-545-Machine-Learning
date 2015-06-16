clear;
data = double(imread('mandrill-small.tiff'));
sz = size(data);
row = sz(1);
col = sz(2);
data = reshape(data,[row * col, sz(3)]);
C = 16;
randix = randi([1,row * col], 1, C);
u = data(randix, :);
label = zeros(row*col);

while(true)
    for i = 1:row * col
        label(i) = find_nearest(data(i,:), u);
    end
    
    %update u
    new_u = zeros(C,3);
    for i = 1:C
        a = data(label == i, :);
        z = size(a);
        if z(1) == 0
            new_u(i,:) = [0,0,0];
        else
            new_u(i,:) = mean(a);
        end
    end
    
    shift = new_u - u;
    err = trace(shift * shift')/3/C;
    if err < 0.01
        break
    end
    u = new_u;
end
%{
trans_data = zeros(row*col, 3);

for i = 1:C
    trans_data(label==i,:) = u(i);
end

trans_data = reshape(trans_data, sz);
image(trans_data);
%}
large = double(imread('mandrill-large.tiff'));
sz = size(large);
N = sz(1)*sz(2);
large = reshape(large, [N, sz(3)]);
new_large = zeros(size(large));
for k = 1:N
    label = find_nearest(large(k,:), u);
    new_large(k,:) = u(label,:);
end
large = large/255;
large = reshape(large,sz);
new_large = new_large/255;
new_large = reshape(new_large,sz);
h = figure('Position', [100, 100, 700, 2100]);
subplot(2,1,1);
title('The compression percentage is 83.3%');
image(new_large);
subplot(2,1,2);
image(large);

print('5','-dpng')
