addpath('../src')
clear;
load('SNAP.mat');
k=100;
s=k/2;

tic;
[U, S, V] = svds(A, 100);
Time_svds = toc;

tic;
[U2, S2, V2] = dashSVD(A, k, 1000, s, 1e-3);
Time_dashSVD = toc;

PVE_dashSVD = max(abs(flipud(diag(U2'*A*(A'*U2))) - Acc_S(1:k).^2))./Acc_S(101).^2;

s = svds(@(x,tflag) Afun(x,tflag,A, U2, diag(S2), V2),size(A),1);
Spec_dashSVD = (s - Acc_S(101))/Acc_S(101);
    
C = A'*U2-V2*diag(S2);
Res_dashSVD = max(sqrt(flipud(diag(C'*C)))./Acc_S(1:k));
    
Sigma_dashSVD = max(abs(flipud(S2)-Acc_S(1:k))./Acc_S(1:k));

semilogy(1:k, diag(S), '+-', 1:k, flipud(S2), 'x-');
axis([0, 101, 0.9*S(100, 100), 1.1*S(1,1)]);
ylabel('\sigma_i');
legend('svds', 'dashSVD');
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',25); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'LineWidth',2); 
set( get(gca,'YAxis'),'LineWidth',2); 
set( get(gca,'Legend'),'FontSize',figure_FontSize); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',1);