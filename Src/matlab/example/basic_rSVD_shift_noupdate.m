function [U, S, V] = basic_rSVD_shift_noupdate(A, k, p, Omg)
    s = k/2;
    [~, n] = size(A);
    l = k+s;
    [Q, ~] = qr(A*Omg, 0);
    alpha = 0;
    for j = 1:p
        [Q, s, ~] = svd(A*(A'*Q)-alpha*Q, 'econ');
        if alpha == 0
            alpha = (alpha + s(l, l))/2;
        end
    end
    B = Q'*A;
    [U, S, V] = svd(B, 'econ');
    U = Q*U(:, 1:k);
    S = S(1:k, 1:k);
    V = V(:, 1:k);
end