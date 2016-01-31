function [inlierindex, outlierindex] = find_points_index(point_list, searching_points, indexPairs)

n = size(searching_points,1);
inlierindex = zeros(n,1);

for ii = 1:n
    d = (point_list(indexPairs,:) - repmat(searching_points(ii,:),size(indexPairs,1),1)).^2;
    d = sum(d,2);
    nearest_points = find(d == min(d));
    inlierindex(ii) = indexPairs(nearest_points(1));
    indexPairs(nearest_points(1)) = [];
end

outlierindex = indexPairs;

return