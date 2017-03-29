close all
m_H_B2A = load('m_H_B2A_1card_test1.mat');
m_H_B2A = m_H_B2A.m_H_B2A;
m_H_A2B = load('m_H_A2B_1card_test1.mat');
m_H_A2B = m_H_A2B.m_H_A2B;

%% ** normalisation **
%for d_f = 1:d_N_f
%    for d_n_meas = 1:d_N_meas
%        m_H_B2A(:,d_n_meas,d_f) = m_H_B2A(:,d_n_meas,d_f)/max(abs(m_H_B2A(:,d_n_meas,d_f)));
%        m_H_A2B(:,d_n_meas,d_f) = m_H_A2B(:,d_n_meas,d_f)/max(abs(m_H_A2B(:,d_n_meas,d_f)));
%    end
%end 
%keyboard;

%% ** average **
%m_H_B2A_ = zeros(size(m_H_B2A,1),15,size(m_H_B2A,3));
%m_H_A2B_ = zeros(size(m_H_A2B,1),15,size(m_H_A2B,3));
%for d_f = 1:d_N_f
%   for d_l = 1:d_N_loc
%       m_H_B2A_(:,d_l,d_f) = mean(m_H_B2A(:,(d_l-1)*10+1:d_l*10,d_f),2); 
%       m_H_A2B_(:,d_l,d_f) = mean(m_H_A2B(:,(d_l-1)*10+1:d_l*10,d_f),2); 
%   end
%end
%keyboard;
%% ** calibration **
for d_f = 1:d_N_f
   [m_F0(:,:,d_f),m_A0_est,m_B0_est] = f_tls_svd(m_H_B2A(:,:,d_f).',m_H_A2B(:,:,d_f).');       
%   [m_F1(:,:,d_f),m_A0_est,m_B0_est] = f_tls_svd(m_H_B2A(:,:,d_f).',m_H_A2B(:,51:100,d_f).');    
 %  [m_F2(:,:,d_f),m_A0_est,m_B0_est] = f_tls_svd(m_H_B2A(:,:,d_f).',m_H_A2B(:,101:150,d_f).');
   [m_F3(:,:,d_f),m_A1_est,m_B1_est] = f_tls_ap(m_H_B2A(:,:,d_f).',m_H_A2B(:,:,d_f).'); 

   %[m_F0(:,:,d_f),m_A0_est,m_B0_est] = f_tls_svd(m_H_B2A_(:,1:50,d_f).',m_H_A2B_(:,1:50,d_f).');       
   %[m_F1(:,:,d_f),m_A0_est,m_B0_est] = f_tls_svd(m_H_B2A_(:,51:100,d_f).',m_H_A2B_(:,51:100,d_f).');    
   %[m_F2(:,:,d_f),m_A0_est,m_B0_est] = f_tls_svd(m_H_B2A_(:,101:150,d_f).',m_H_A2B_(:,101:150,d_f).');
   %[m_F3(:,:,d_f),m_A1_est,m_B1_est] = f_tls_ap(m_H_B2A_(:,:,d_f).',m_H_A2B_(:,:,d_f).'); 
end
%oarf_stop(cardA);
%oarf_stop(cardB);

%% ** plot **
figure(10)
hold on;
for d_f=1:size(m_F0,3);
  m_F= m_F0(:,:,d_f);
  plot(m_F(1,1),'bo')
  plot(m_F(2,2),'ko')
  plot(diag(m_F,1),'r+')
  plot(diag(m_F,-1),'gx')
end
hold off;
title('F0');
axis([-10 10 -10 10])
grid on

figure(11)
hold on;
for d_f=1:size(m_F1,3);
  m_F= m_F1(:,:,d_f);
  plot(m_F(1,1),'bo')
  plot(m_F(2,2),'ko')
  plot(diag(m_F,1),'r+')
  plot(diag(m_F,-1),'gx')
end
hold off;
title('F1');
axis([-10 10 -10 10])
grid on

figure(12)
hold on;
for d_f=1:size(m_F2,3);
  m_F= m_F2(:,:,d_f);
  plot(m_F(1,1),'bo')
  plot(m_F(2,2),'ko')
  plot(diag(m_F,1),'r+')
  plot(diag(m_F,-1),'gx')
end
hold off;
title('F2');
axis([-10 10 -10 10])
grid on;

figure(13)
hold on;
for d_f=1:size(m_F3,3);
  m_F= m_F3(:,:,d_f);
  plot(m_F(1,1),'bo')
  plot(m_F(2,2),'ko')
  plot(diag(m_F,1),'r+')
  plot(diag(m_F,-1),'gx')
end
hold off;
title('F3');
axis([-10 10 -10 10])
grid on
