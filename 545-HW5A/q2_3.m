clear;
data = double(imread('mandrill-small.tiff'));
sz = size(data);
row = sz(1);
col = sz(2);
N = row * col;
dim = sz(3);
data = reshape(data,[N, dim]);
C = 16;
total = N;
randix = randi([1,row * col], 1, C);
u = data(randix, :);
prior = ones([C,1])/C;
posteri = zeros([N, C]);
cov = zeros([C,dim,dim]);
for i = 1:C
    cov(i,:,:) = eye(dim);
end
detm = ones(C,1);
inv_covs = zeros([C,dim,dim]);
log_like = 0;
count = 0;
while (true)
    count = count+1;
    for i = 1:C
        sigma = reshape(cov(i,:,:), [dim, dim]);
        detm(i) = det(sigma);
        inv_covs(i,:,:) = inv(sigma);
    end
    for i = 1:N
        posteri(i,:) = comp_posteri(data(i,:), u, detm, inv_covs, prior);
    end
    
    M = sum(posteri);
    new_u = posteri' * data;
    for i = 1:C
        new_u(i,:) = new_u(i,:)/M(i);
    end
    
    for i = 1:C
        temp = bsxfun(@minus, data, new_u(i));
        cov(i,:,:) = temp' * diag(posteri(:,i)) * temp/M(i);
    end
    
    prior = M/N;
    shift = new_u - u;
    err = trace(shift * shift')/3/C
    if err < 0.01
        break
    end
    u = new_u;
    
    for i = 1:C
        sigma = reshape(cov(i,:,:), [dim, dim]);
        detm(i) = det(sigma);
        inv_covs(i,:,:) = inv(sigma);
    end
    log_like = 0;
    const = sqrt(2*pi)^3;
    for i = 1:N
        temp = 0;
        for c = 1:C
            sigma = reshape(cov(c,:,:), [dim, dim]);
            mu = u(c,:);
            temp = temp + prior(c)/sqrt(detm(c))/const * exp(-(data(i,:)-mu)*reshape(inv_covs(c,:,:),[3,3])*(data(i,:)-mu)'/2);
        end
        log_like = log_like + log(temp);
    end
    log_like
end


data = double(imread('mandrill-large.tiff'));
sz = size(data);
row = sz(1);
col = sz(2);
N = row * col;
dim = sz(3);
data = reshape(data,[N, dim]);
new_data = zeros(size(data));
C = 16;

for i = i : N
    nearest = 1;
    min_loglike = 0;
    for c = 1:C
        loglike = log(prior(c)) - log(detm(c))/2 - (data(i,:)-u(c,:))*reshape(inv_covs(c,:,:),[3,3])*(data(i,:)-u(c,:))'/2;
        if c == 1 | loglike > min_loglike
            nearest = c;
            min_loglike = loglike;
        end
        
    end
    new_data(i,:) = u(nearest,:);
end

data = data/255;
data = reshape(data,sz);
new_data = new_data/255;
new_data = reshape(new_data,sz);
h = figure('Position', [100, 100, 700, 3000]);
subplot(2,1,1);
image(new_data);
title(strcat('Avg log likelihood = ', num2str(log_like/total)));
subplot(2,1,2);
image(data);
print('6','-dpng');


