function M = gaussian(coord, e)
%
%   Exponential Kernel matrix, K(x,y) = exp(-e * ||x-y||^2)
%
if isnumeric(coord)
    N = size(coord, 1);
    if N == 1
        M = 1;
        return ;
    end 
    dist = pdist(coord);
    M = squareform(exp(- e * dist.^2)) + eye(N);
elseif iscell(coord)&&length(coord)==2      
    M = pdist2(coord{1}, coord{2});
    M = exp(- e * M.^2);
end
end