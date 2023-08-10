clear;
% For matrix Dense1
load('Dense1');
pmax = 24;

% For matrix Dense2
% load('Dense2');
% pmax = 14;

[U, S, V] = svd(A);
ss = diag(S(1:100, 1:100)).^2;
s101 = S(101, 101).^2;

err1 = [];
err2 = [];
err3 = [];

k = 100;
for p = 0:2:pmax
    [u1, s1, v1] = basic_rSVD(A, k, p, Omega);
    sst = diag((u1'*A)*A'*u1);
    pvet = max(abs(sst-ss)./s101);
    err1 = [err1, pvet];
    
    [u2, s2, v2] = basic_rSVD_shift_noupdate(A, k, p, Omega);
    sst = diag((u2'*A)*A'*u2);
    pvet = max(abs(sst-ss)./s101);
    err2 = [err2, pvet];
    
    [u3, s3, v3] = basic_rSVD_shift(A, k, p, Omega);
    sst = diag((u3'*A)*A'*u3);
    pvet = max(abs(sst-ss)./s101);
    err3 = [err3, pvet];
end

p=pmax;
semilogy(0:2:p, err1, 'o-', 0:2:p, err2, '^-', 0:2:p, err3, 's-');
xlabel('\itp')
ylabel('\epsilon_{PVE}');
legend('Alg. 1', 'Alg. 2^*', 'Alg. 2', 'Location', 'Southwest');
xmin = 0;
xmax = p+1;
ymin = 0.8*min(err3);
ymax = 2*err1(1);
axis([xmin, xmax, ymin, ymax]);
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',25); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'LineWidth',2); 
set( get(gca,'YAxis'),'LineWidth',2); 
set( get(gca,'Legend'),'FontSize',figure_FontSize); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2);
set(gca, 'YTick', [1e-6, 1e-4, 1e-2, 1]);