function sample_pt = H2_select_sample(htree, approx_rank, alpha)
% For each node, select sample points for ALL its far-field points
% Input parameters:
%   htree       : Hierarchical partitioning tree structure
%   approx_rank : Approximation rank on each dimension
%   alpha       : Admissible criteria, 1 == H2, 0 == HSS
% Output parameters:
%   sample_pt : Size htree.nnode, sample points of each node

    [~, far_pairs] = H2__block_partition_box(htree, alpha);
    min_lvl   = min(htree.nodelvl(far_pairs(:)));
    max_lvl   = htree.nlevel;
    lvl_nodes = htree.level;
    n_node    = htree.nnode;
    children  = htree.children;
    coord     = htree.coord;
    cluster   = htree.cluster;
    pt_dim    = size(htree.enbox{1}, 2);
    far_list  = cell(n_node, 1);
    for i = 1 : size(far_pairs, 1)
        c1 = far_pairs(i, 1);
        c2 = far_pairs(i, 2);
        far_list{c1} = [far_list{c1}; c2];
        far_list{c2} = [far_list{c2}; c1];
    end

    %% Bottom-up sweep
    if (approx_rank >= 15), ri = approx_rank - 9; end
    if (approx_rank <= 14), ri = approx_rank - 8; end
    if (approx_rank <=  9), ri = approx_rank - 5; end
    if (approx_rank <=  7), ri = approx_rank - 4; end
    if (approx_rank <=  3), ri = approx_rank - 1; end
    if (ri < 0), ri = 0; end
    clu_refine = cell(n_node, 1);
    for i = max_lvl : -1 : min_lvl
        % Update refined points associated with the clusters at i-th level
        for j = 1 : length(lvl_nodes{i})
            node = lvl_nodes{i}(j);
            child_node = children(node, ~isnan(children(node, :)));
            if (isempty(child_node))
                clu_s = cluster(node, 1);
                clu_e = cluster(node, 2);
                clu_refine{node} = coord(clu_s : clu_e, :);
            else
                clu_refine{node} = vertcat(clu_refine{child_node});
            end
        end

        % Select refined points for all nodes at i-th level
        for j = 1 : length(lvl_nodes{i})
            node = lvl_nodes{i}(j);
            npt_refine = size(clu_refine{node}, 1);
            if (npt_refine <= 5), continue; end
            % Volume sampling, estimate ri_vol = number of sample points in this node
            node_enbox_size = max(clu_refine{node}) - min(clu_refine{node});
            node_enbox_size = node_enbox_size(1 : pt_dim);
            [ri_list, ri_vol] = proportional_decompose(node_enbox_size, ri);
            if (ri_vol < npt_refine)
                sample_idx = H2_select_cluster_sample(pt_dim, clu_refine{node}, ri_list, 6);
                clu_refine{node} = clu_refine{node}(sample_idx, :);
            end
        end
    end

    %% Top-down sweep
    sample_pt = cell(n_node, 1);
    if (approx_rank > 3)
        ri = max(approx_rank + 3, 10);
    else
        ri = approx_rank + 3;
    end
    for i = min_lvl : max_lvl
        for j = 1 : length(lvl_nodes{i})
            node = lvl_nodes{i}(j);
            if (isempty(far_list{node})), continue; end

            % Gather all far-field refined points as initial sample points
            sample_pt{node} = [sample_pt{node}; vertcat(clu_refine{far_list{node}})];

            % Refine initial sample points
            sample_enbox_size = max(sample_pt{node}) - min(sample_pt{node});
            sample_enbox_size = sample_enbox_size(1 : pt_dim);
            [ri_list, ri_vol] = proportional_decompose(sample_enbox_size, ri);
            n_sample = size(sample_pt{node}, 1);
            if (ri_vol < n_sample)
                sample_idx = H2_select_cluster_sample(pt_dim, sample_pt{node}, ri_list, 2);
                sample_pt{node} = sample_pt{node}(sample_idx, :);
            end

            % Pass refined sample points to children
            child_node = children(node, ~isnan(children(node, :)));
            for k = 1 : length(child_node)
                child_k = child_node(k);
                sample_pt{child_k} = sample_pt{node};
            end
        end
    end
end

function [p_list, n_node] = proportional_decompose(prop, p_total)
% Decompose p_total = sum(p_list) && length(p_list) == length(prop) s.t.
% p_list(i) ~ prop(i) / sum(prop) * p_total
    p_list_prop = p_total .* prop ./ sum(prop);
    p_list = floor(p_list_prop);
    while (sum(p_list) < p_total)
        [~, idx] = max(p_list_prop - p_list);
        % Add 1 to positions that got hit most by floor
        for i = 1 : min(length(idx), p_total - sum(p_list))
            p_list(idx(i)) = p_list(idx(i)) + 1;
        end
    end
    n_node = prod(1 + p_list);
end