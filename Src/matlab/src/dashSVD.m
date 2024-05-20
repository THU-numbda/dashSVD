function [U, S, V] = dashSVD(A, k, p, s, tol)
% this is the fast randomized PCA for sparse data
if nargin < 5
    tol = 1e-2;
end
if nargin < 4
    s = round(k/2);
end
if nargin < 3
    p = 1000;
end
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
    sk = zeros(l, 1);
    sk_now = 0;
    for i = 1:p
        [Q, S, ~] = eigSVD(A'*(A*Q)-alpha*Q);
        sk_now = S+alpha;
        pve_all = abs(sk_now-sk)./sk_now(s);
        ei = max(pve_all(s+1:k));
        if ei < tol
            break
        end
        if alpha < S(1)
            alpha = (alpha+S(1))/2;
        end
        sk = sk_now;
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
    sk = zeros(l, 1);
    sk_now = 0;
    for i = 1:p
        [Q, S, ~] = eigSVD(A*(A'*Q)-alpha*Q);
        sk_now = S+alpha;
        pve_all = abs(sk_now-sk)./sk_now(s);
        ei = max(pve_all(s+1:k));
        if ei < tol
            break
        end
        if alpha < S(1)
            alpha = (alpha+S(1))/2;
        end
        sk = sk_now;
    end
    [V,S,U] = eigSVD(A'*Q);
    ind = s+1:k+s;
    U = Q*U(:, ind);
    V = V(:, ind);
    S = S(ind);
end
end
