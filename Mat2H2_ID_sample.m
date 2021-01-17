function h2mat = Mat2H2_ID_sample(kernel, htree, sample_pt, type, par, alpha, JIT_flag)
% Construction a symmetric H2 representation with ID and far-field sample points
% Input parameters:
%   kernel    : Handle of the kernel function
%   htree     : Hierarchical partitioning tree structure
%   sample_pt : Size htree.nnode, sample points obtained from H2_select_sample()
%   type, par : ID approximation parameters (QR stop type and QR stop parameter)
%   alpha     : Admissible criteria, 1 == H2, 0 == HSS
%   JIT_flag  : If we need to compute & store H2 representation B & D matrices
% Output parameter:
%   h2mat : Symmetric H2 representation data structure

    if nargin < 7
        JIT_flag = false;
    end

    % Basic info
    parent   = htree.parent;
    children = htree.children;
    level    = htree.level;
    nodelvl  = htree.nodelvl;
    leafnode = htree.leafnode;
    enbox    = htree.enbox;
    cluster  = htree.cluster;
    nlevel   = htree.nlevel;
    coord    = htree.coord;
    nnode    = htree.nnode;
    ptdim    = size(enbox{1}, 2);
    kdim     = size(kernel(randn(2, ptdim)), 1) / 2;

    % Get all admissible and inadmissible node pairs
    [near, far] = H2__block_partition_box(htree, alpha);
    minlvl = min(nodelvl(far(:)));

    % Allocate cell arrays
    U = cell(nnode, 1);
    I = cell(nnode, 1);
    B = cell(nnode);
    D = cell(nnode);
    if (isscalar(par))
        par = par * ones(nlevel, 1);
    end
    if (strcmp(type, 'reltol'))
        par = par * 1e-2;
    end

    %% Hierarchical construction U level by level
    for i = nlevel : -1 : minlvl
        % Update skeleton points associated with the clusters at i-th level
        for j = 1 : length(level{i})
            node = level{i}(j);
            child_node = children(node, ~isnan(children(node, :)));
            if isempty(child_node) 
                I{node} = cluster(node, 1) : cluster(node, 2);
            else
                I{node} = horzcat(I{child_node});
            end 
        end
        
        % Compression all nodes at i-th level
        for j = 1 : length(level{i})
            node = level{i}(j);
            node_ff_coord = sample_pt{node};
            skel_coord = coord(I{node}, :);
            A_sample = kernel({skel_coord, node_ff_coord});
            if (kdim == 1)
                [U{node}, subidx] = ID(A_sample, type, par(i));
            else
                [U{node}, subidx] = IDdim(A_sample, kdim, type, par(i));
                subidx = subidx(kdim:kdim:end)/kdim;
            end
            I{node} = I{node}(subidx);     
        end    
    end

    %% Build B & D matrices
    if JIT_flag == false
        % D matrices: diagonal blocks   
        for i = 1 : length(leafnode)
            node = leafnode(i);
            idx  = cluster(node,1 ) : cluster(node, 2);
            D{node, node} = kernel(coord(idx, :));
        end

        % D matrices: off-diagonal blocks for inadmissible pairs
        for i = 1 : size(near, 1)
            c1   = near(i, 1);
            c2   = near(i, 2);
            idx1 = cluster(c1, 1) : cluster(c1, 2);
            idx2 = cluster(c2, 1) : cluster(c2, 2);
            D{c1, c2} = kernel({coord(idx1, :), coord(idx2, :)});
        end   

        % B matrices: intermediate blocks for admissible pairs
        for i = 1 : size(far, 1)
            c1   = far(i, 1);
            c2   = far(i, 2);
            idx1 = cluster(c1, 1) : cluster(c1, 2);
            idx2 = cluster(c2, 1) : cluster(c2, 2);
            if nodelvl(c1) == nodelvl(c2)
                % c1 & c2 are on the same level, compress on both sides
                B{c1,c2} = kernel({coord(I{c1}, :), coord(I{c2}, :)}); 
            elseif nodelvl(c1) > nodelvl(c2)  
                % c2 is a leaf node at higher level, only compress on c1 side
                B{c1,c2} = kernel({coord(I{c1}, :), coord(idx2,  :)});
            else
                % c1 is a leaf node at higher level, only compress on c2 side
                B{c1,c2} = kernel({coord(idx1,  :), coord(I{c2}, :)});
            end
        end
    end

    %% Wrap up
    h2mat.U = U;
    h2mat.I = I;
    h2mat.B = B;
    h2mat.D = D;
    h2mat.near      = near;
    h2mat.far       = far;
    h2mat.minlvl    = minlvl;
    h2mat.JIT       = JIT_flag;
    h2mat.alpha     = alpha;
    h2mat.type      = type;
    h2mat.par       = par;
    h2mat.kernel    = kernel;
    h2mat.kdim      = kdim;
    h2mat.sample_pt = sample_pt;
    h2mat.storage   = H2__storage_cost(h2mat, htree);
    h2mat.rankinfo  = H2__rank(h2mat, htree);
end