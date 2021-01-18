function sample_idx = H2_select_cluster_sample(pt_dim, coord, grid_size, algo)
% For each point in the generated anchor grid, choose O(1) nearest points 
% in the given point cluster as sample points
% Input parameters:
%   pt_dim    : Point dimension
%   coord     : Size npt * dim, target point cluster, each row is a point coordinate
%   grid_size : Size dim, size of the anchor grid (number of anchor points in each dimension)
%   algo      : Anchor grid generation algorithm
% Output parameter:
%   sample_idx : Size unknown column vector, ranged [1, npt], indices of chosen sample points

    grid_size1 = grid_size + 1;
    npt_anchor = prod(grid_size1);
    coord_max  = max(coord);
    coord_min  = min(coord);
    enbox_size = coord_max - coord_min;

    %% Assign anchor points in each dimension
    anchor_dim = cell(1, pt_dim);
    if (min(grid_size) == 0)
        zero_list = (grid_size == 0);
        anchor_dim(zero_list) = num2cell((coord_max(zero_list) + coord_min(zero_list)) / 2);
    end
    idx = 1 : pt_dim;
    idx = idx(grid_size > 0);
    if (algo == 2)
        % Chebyshev anchor points
        for i = 1 : length(idx)
            k = idx(i);
            s0 = (coord_max(k) + coord_min(k)) / 2;
            s1 = (coord_max(k) - coord_min(k)) / 2;
            s2 = pi / (2 * grid_size(k) + 2);
            v0 = 2 * (0 : grid_size(k))' + 1;
            v1 = cos(v0 * s2);
            anchor_dim{k} = s0 + s1 * v1;
        end
    end
    if (algo == 6)
        c0 = 1.0;
        c1 = 0.5;
        c2 = 0.25;
    end
    if (algo >= 4)
        for i = 1 : length(idx)
            k = idx(i);
            size_k = c0 * enbox_size(k) / (grid_size(k) + c1);
            anchor_dim{k} = coord_min(k) + c2 * size_k + size_k * (0 : grid_size(k))';
        end
    end

    %% Do a tensor product to get all anchor points
    anchor_coord = zeros(npt_anchor, pt_dim);
    stride = [1 cumprod(grid_size1)];
    for idx = 1 : npt_anchor
        for i_dim = 1 : pt_dim
            dim_idx = mod(floor(idx / stride(i_dim)), grid_size1(i_dim)) + 1;
            anchor_coord(idx, i_dim) = anchor_dim{i_dim}(dim_idx);
        end
    end

    %% Choose nearest points in the given point cluster
    m = 1;
    [~, idx] = pdist2(coord(:, 1 : pt_dim), anchor_coord, 'euclidean', 'Smallest', m);
    sample_idx = unique(idx(:));
end