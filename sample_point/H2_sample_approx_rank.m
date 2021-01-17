function r = H2_sample_approx_rank(tau, tol)
% Determine the order of Taylor expansion (approximation rank)
% This function is copied from determinedRank3D(), not sure if it
% can be used for other dimensions
    if (tol < 7e-9)
        r = ceil( sqrt( max( 2*floor(log(tol)/log(tau)), 90 ) ) );
    elseif (tol < 7e-7)
        r = ceil( sqrt( max( 2*floor(log(tol)/log(tau)-10), 20) ) );
    elseif (tol < 2e-4)
        r = ceil( sqrt( max( 2*floor(log(tol)/log(tau)-15), 20) ) );
    elseif (tol < 2e-3)
        r = 4;
    elseif (tol < 2e-2)
        r = 3;
    elseif (tol < 2e-1)
        r = 2;
    else
        r = 1;
    end
end