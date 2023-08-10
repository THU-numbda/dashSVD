function [U, S, V] = dashSVD_origin(A, k, p, s)
% this is the fast randomized PCA for sparse data
if p < 0
    warning('Power parameter p must be no less than 0 !');
    return;
end
[m, n] = size(A);
l = k+s;

%This is designed for m >= n
if (m>=n)
    Q = randn(m, l);
    Q = A'*Q;
    [Q, ~, ~] = eigSVD(Q);
    alpha = 0;
    for i = 1:p
        [Q, S, ~] = eigSVD(A'*(A*Q)-alpha*Q);
        if alpha < S(1)
            alpha = (alpha+S(1))/2;
        end
    end
    [U,S,V] = eigSVD(A*Q);
    ind = s+1:k+s;
    U = U(:, ind);
    V = Q*V(:, ind);
    S = S(ind);

%This is designed for matrix m<n
else
    Q = randn(n, l);
    Q = A*Q;
    [Q, ~, ~] = eigSVD(Q);
    alpha = 0;
    for i = 1:p
        [Q, S, ~] = eigSVD(A*(A'*Q)-alpha*Q);
        if alpha < S(1)
            alpha = (alpha+S(1))/2;
        end
    end
    [V,S,U] = eigSVD(A'*Q);
    ind = s+1:k+s;
    U = Q*U(:, ind);
    V = V(:, ind);
    S = S(ind);
    end
end
