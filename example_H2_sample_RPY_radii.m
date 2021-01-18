%% Problem parameters
eta    = 1;
kernel = @(coord)rpy_mex_general(coord, eta);
npt    = 10000;
ptdim  = 3;
kdim   = 3;
rng(19241112);
coord  = (4/3*pi * npt / 0.4)^(1/ptdim) * rand(npt, ptdim);
radii  = 0.5 + 5 * rand(npt, 1);

%% 1. Hierarchical partitioning 
max_leaf_npt = 200;
htree = hierarchical_partition(coord, max_leaf_npt, ptdim);
coord = [coord, radii];
radii(htree.permutation) = radii;
htree.coord = [htree.coord, radii];

%% 2. Select sample points
alpha  = 1;
tau    = 0.7;
reltol = 1e-8;
approx_rank = H2_sample_approx_rank(tau, reltol);
sample_pt   = H2_select_sample(htree, approx_rank, alpha);

%% 3. H2 matrix construction using proxy points
JIT_flag = true;
tic;
h2mat = Mat2H2_ID_sample(kernel, htree, sample_pt, 'reltol', reltol, alpha, JIT_flag);
h2_build_t = toc;
fprintf('H2 construction time = %.3f\n', h2_build_t);

% Calculate rank
max_rank = 0;
avg_rank = 0;
node_cnt = 0;
for i = 1 : htree.nnode
    node_rank = size(h2mat.U{i}, 2);
    if (node_rank > 0)
        max_rank = max(max_rank, node_rank);
        avg_rank = avg_rank + node_rank;
        node_cnt = node_cnt + 1;
    end
end
avg_rank = ceil(avg_rank / node_cnt);
fprintf('Average/max rank = %d, %d\n', avg_rank, max_rank);

%% 4. H2 matrix-vector multiplications
n_test = 10;
x      = rand(npt * kdim, n_test) - 0.5;
tic;
u_h2   = H2_matvec(h2mat, htree, x);
h2_matvec_t = toc;
idx    = 1 : 1000;
u_ref  = kernel({coord(idx, :), coord}) * x;
u_h2_p = u_h2(1 : 1000 * kdim, :);
relerr = zeros(n_test, 1);
for i = 1 : n_test
    relerr = norm(u_h2_p(:, i) - u_ref(:, i)) / norm(u_ref(:, i));
end
fprintf('%d H2 matvec time = %.3f\n', n_test, h2_matvec_t);
fprintf("min/mean/max relative errors for %d matvec:\n%.3e,%.3e,%.3e\n", ...
    n_test, min(relerr), mean(relerr), max(relerr));