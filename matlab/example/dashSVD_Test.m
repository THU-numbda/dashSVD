clear;
maxNumCompThreads(1);
load('SNAP.mat');


k=100;
s=k/2;
[m, n] = size(A);

pve_svds=[];
spec_svds=[];
res_svds=[];
sigma_svds=[];
time_svds=[];


for i=1:1:5
    opts.p = 150;
    opts.tol = 1e-10;
    opts.disp = 0;
    opts.maxit = i;
    
    tic;
    [U2, S2, V2] = svdstest(A, 100, 'largest', opts);
    time_temp = toc
    time_svds = [time_svds; time_temp];
    
    pve = max(abs((diag(U2'*A*(A'*U2))) - Acc_S(1:k).^2))./Acc_S(101).^2;
    pve_svds = [pve_svds; pve];
    
    s = svds(@(x,tflag) Afun(x,tflag,A, U2, S2, V2),size(A),1);
    spec = (s - Acc_S(101))/Acc_S(101);
    if spec <= 0
        spec = 1e-16;
    end
    spec_svds = [spec_svds; spec];
    
    C = A'*U2-V2*S2;
    res = max((sqrt(diag(C'*C)))./Acc_S(1:k));
    res_svds = [res_svds; res];
    
    sigma = max(abs(diag(S2)-Acc_S(1:k))./Acc_S(1:k));
    sigma_svds = [sigma_svds; sigma];
    clear('C', 'U2', 'V2');
end

pve_rsvd=[];
spec_rsvd=[];
res_rsvd=[];
sigma_rsvd=[];
time_rsvd=[];


for i=0:4:32
    tic;
    [U2, S2, V2] = dashSVD_origin(A, k, i, 0.5*k);
    time_temp = toc
    time_rsvd = [time_rsvd; time_temp];
    
    pve = max(abs(flipud(diag(U2'*A*(A'*U2))) - Acc_S(1:k).^2))./Acc_S(101).^2;
    pve_rsvd = [pve_rsvd; pve];
    
    s = svds(@(x,tflag) Afun(x,tflag,A, U2, diag(S2), V2),size(A),1);
    spec = (s - Acc_S(101))/Acc_S(101);
    spec_rsvd = [spec_rsvd;spec];
    
    C = A'*U2-V2*diag(S2);
    res = max(sqrt(flipud(diag(C'*C)))./Acc_S(1:k));
    res_rsvd = [res_rsvd; res];
    
    sigma = max(abs(flipud(S2)-Acc_S(1:k))./Acc_S(1:k));
    sigma_rsvd = [sigma_rsvd;sigma];

    clear('C', 'U2', 'V2');
end

figure(1)
semilogy(time_svds, pve_svds, '+-', time_rsvd, pve_rsvd, 'o-');
legend('LanczosBD', 'dashSVD', 'Location', 'Southwest');
ylabel('\epsilon_{PVE}');
xlabel('Time');
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',20); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 

figure(2)
semilogy(time_svds, spec_svds, '+-', time_rsvd, spec_rsvd, 'o-');
legend('LanczosBD', 'dashSVD', 'Location', 'Southwest');
ylabel('\epsilon_{spec}');
xlabel('Time');
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',20); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 

figure(3)
semilogy(time_svds, sigma_svds, '+-', time_rsvd, sigma_rsvd, 'o-');
legend('LanczosBD', 'dashSVD', 'Location', 'Southwest');
ylabel('\epsilon_{\sigma}');
xlabel('Time');
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',20); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 

figure(4)
semilogy(time_svds, res_svds, '+-', time_rsvd, res_rsvd, 'o-');
legend('LanczosBD', 'dashSVD', 'Location', 'Southwest');
ylabel('\epsilon_{res}');
xlabel('Time');
figure_FontSize=25; 
set(findobj('FontSize',10),'FontSize',20); 
set( get(gca,'XLabel'),'FontSize',figure_FontSize); 
set( get(gca,'YLabel'),'FontSize',figure_FontSize); 
set( get(gca,'XAxis'),'FontSize',figure_FontSize); 
set( get(gca,'YAxis'),'FontSize',figure_FontSize); 
set(findobj( get(gca,'Children'),'LineWidth',0.5),'LineWidth',2); 
