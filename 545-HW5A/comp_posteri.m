function post = comp_posteri(data, u, detm, inv_covs, prior)
%POSTERI Summary of this function goes here
%   Detailed explanation goes here
C = length(prior);
post = zeros(C,1);
log_like = zeros(C,1);
for k = 1:C
    mu = u(k,:);
    inv_cov = reshape(inv_covs(k,:,:),[3,3]);
    log_like(k) = log(prior(k)/sqrt(detm(k))) -(data-mu)*inv_cov*(data-mu)'/2;
end

max_log = max(log_like);
for k = 1:C
    post(k) = exp(log_like(k)-max_log);
end
post = post/sum(post);

end

