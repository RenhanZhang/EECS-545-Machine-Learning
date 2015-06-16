function label = find_nearest(v, m)
%FIND_NEAREST Summary of this function goes here
%   Detailed explanation goes here
label = 0;
min_dist = 0;

for i = 1:length(m)
    residual = m(i,:)-v;
    
    dist = residual*residual';
    if i == 1 | dist < min_dist
        label = i;
        min_dist = dist;
    end
end  

end

