function [U, S, V] = basic_rSVD(A, k, p, Omg)
    s = k/2;
    [~, n] = size(A);
    l = k+s;
    [Q, ~] = qr(A*Omg, 0);
    for j = 1:p
        [Q, ~] = qr(A*(A'*Q), 0);
    end
    B = Q'*A;
    [U, S, V] = svd(B, 'econ');
    U = Q*U(:, 1:k);
    S = S(1:k, 1:k);
    V = V(:, 1:k);
end