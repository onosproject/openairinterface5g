%
% SCRIPT ID : s_run_calib
%
% PROJECT NAME : TDD Recoprocity
%
% PURPOSE : channel calibration for MISO case
%
%**********************************************************************************************
%                            Eurecom -  All rights reserved
%
% AUTHOR(s) : Xiwen JIANG, Florian Kaltenberger
%
% DEVELOPMENT HISTORY :
%
% Date         Name(s)       Version  Description
% -----------  ------------- -------  ------------------------------------------------------
% Apr-30-2014  X. JIANG       0.1     script creation v0.1
% Feb-21-2017  X. JIANG       0.2     script adaption for WSA demo at Berlin
%
% REFERENCES/NOTES/COMMENTS :
%
% - Based on the script "calibration" created by Mirsad Cirkic, Florian Kaltenberger.
% 
%**********************************************************************************************

%% ** initilisation **
%---------- to change in experiement --------- 
%clc
%clear all
%close all
%d_N_f = 301; 			% carrier number carrying data
%d_N_meas = 10;          % measuement number
%d_N_loc = 5;            % Rx locations
%d_N_antM = 2;           % max active antenna number for site a and site b
%----------------------------------------------

%% -------- System parameters --------
d_M = 4;			% modulation order, e.g. 4 means QPSK

%** frequency **
d_N_f = 301; 			% carrier number carrying data
d_N_FFT = 512;			% total carrier number
d_N_CP = 128;			% extented cyclic prefix
%** time **
d_N_OFDM = 120;			% number of ofdm symbol per frame
d_N_meas = 1;			% measuement number
%** space **
d_N_antA = 4;			% antenna number at site a
d_N_antB = 4; 			% antenna number at site b
v_indA = find(v_active_rfA);	% active antenna index at site a
v_indB = find(v_active_rfB);	% active antenna index at site b
%** amplitude **
d_amp = pow2(13)-1;   % to see how to be used??

%% -------- calibration parameters -------
d_N_loc = 1;            % Rx locations
d_N_antM = max(sum(v_active_rfA),sum(v_active_rfB));           % max active antenna number for site a and site b

m_H_A2B = zeros(d_N_antM,d_N_meas*d_N_loc, d_N_f);           % d_N_antA x (d_N_meas*d_N_loc) x d_N_f
m_H_B2A = zeros(d_N_antM,d_N_meas*d_N_loc, d_N_f);           % d_N_antA x (d_N_meas*d_N_loc) x d_N_f

m_F = zeros(d_N_antM,d_N_f);
m_F_ = zeros(d_N_antM,d_N_meas,d_N_f);

%% ** collect the measurement data from different locations **
d_loc = 1;
while(d_loc <= d_N_loc)
    % run measurement, note: uncomment "clear all"
    s_run_meas;                                   
    % -----------------------------------------------------
    d_yes = yes_or_no('valid measurement?');    
    if d_yes == 1
      m_H_A2Bi = permute(squeeze(m_H_est_A2B),[1 3 2]);
      m_H_B2Ai = permute(squeeze(m_H_est_B2A),[1 3 2]);
      m_H_A2B(:,(d_loc-1)*d_N_meas+1:d_loc*d_N_meas,:) = m_H_A2Bi;
      m_H_B2A(:,(d_loc-1)*d_N_meas+1:d_loc*d_N_meas,:) = m_H_B2Ai;
      d_loc = d_loc +1
    end   
    %keyboard;
    pause
end
%s_run_meas;                                   
%% --- the following part is dedicated to B2A MISO -----
%m_H_A2B = squeeze(m_H_est_A2B);
%m_H_B2A = squeeze(m_H_est_B2A);
%% -----------------------------------------------------

%% ** calibration **
for d_f = 1:d_N_f
  m_F(:,d_f) = mean(m_H_A2B(:,:,d_f)./m_H_B2A(:,:,d_f),2);
end

m_F_norm = zeros(d_N_antM+1, d_N_f);
m_F_norm(1, :) = mean(m_F(1, :),2);
m_F_norm(2, :) = mean(m_F(2, :),2);
m_F_norm(3, :) = 1;
m_F_norm = m_F_norm./max(max(abs(m_F_norm)))*0.99;

%keyboard

save('-v7','result/m_F.mat','m_F'); 
%% ** transform the data to Q2.14 format and store it in a .mtx file
m_F_Q15 = zeros(d_N_antM+1,d_N_f*2); 
m_F_Q15(:,1:2:end-1) = floor(real(m_F_norm)*(2^15));
m_F_Q15(:,2:2:end) = floor(imag(m_F_norm)*(2^15));
%%save('-ascii','calibF.mtx','m_F2_diag_Q14');
dlmwrite('result/calibF.mtx', m_F_Q15,' ');

%% ** plot **
figure(11)
hold on;
for d_f=1:d_N_f
  plot(m_F_norm(1,d_f),'bo')
  plot(m_F_norm(2,d_f),'ro')
  plot(m_F_norm(3,d_f)+0.000001*1i,'ko')
end
hold off;
title('Diagonal F');
axis([-2 2 -2 2])
grid on

%figure(12)
%hold on;
%for d_f=1:d_N_f
%  plot(m_F_(1,1,d_f),'bo')
%  plot(m_F_(2,1,d_f),'ro')
%%  plot(m_F_(1,3,d_f),'gx')
%%  plot(m_F_(2,3,d_f),'yx')
%%  plot(m_F_(1,5,d_f),'c+')
%%  plot(m_F_(2,5,d_f),'m+')
%end
%hold off;
%title('Diagonal F');
%axis([-2 2 -2 2])
%grid on
