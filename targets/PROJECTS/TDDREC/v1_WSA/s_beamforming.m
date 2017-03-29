%
% SCRIPT ID : s_beamforming
%
% PROJECT NAME : TDD Recoprocity
%
% PURPOSE : perform beamforming based on TDD calibration
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
%
% REFERENCES/NOTES/COMMENTS :
%
% - Based on the script "beamforming" created by Mirsad Cirkic, Florian Kaltenberger.
% 
%**********************************************************************************************
clear all

%% -------- initilisation --------
d_M = 4;			% modulation order, e.g. 4 means QPSK

%** frequency **
d_fc  = 2580000000; %1907600000; 
d_delta_f = 15000;
d_N_f = 301; 			% carrier number carrying data
d_N_FFT = 512;			% total carrier number
d_N_CP = 128;			% extented cyclic prefix
%%** time **
d_N_OFDM = 120;			% number of ofdm symbol per frame
d_N_meas = 1;			% measuement number
%%** space **
d_N_antA = 4;			% antenna number at site a
d_N_antB = 4; 			% antenna number at site b
v_active_rfA = [1 1 0 0]; 
v_active_rfB = [0 0 0 1];
active_rf = v_active_rfA | v_active_rfB;
v_indA = find(v_active_rfA);	% active antenna index at site a
v_indB = find(v_active_rfB);	% active antenna index at site b
d_N_antA_act = length(v_indA);
d_N_antB_act = length(v_indB);
%
%%** amplitude **
d_amp = pow2(13);  

d_n_bit = 16;
card = 0;
%% -------- load F -------- 
o_result = load('result/m_F.mat');
m_F = o_result.m_F;

%% -------- channel measurement -------- 
s_run_meas;

%% -------- signal precoding -------- 
v_MPSK = exp(sqrt(-1)*([1:d_M]*2*pi/d_M+pi/d_M));						
m_sym_TA = v_MPSK(ceil(rand(d_N_f, d_N_OFDM)*d_M));   

m_sym_TA_ideal = zeros(d_N_f,d_N_OFDM,d_N_antA_act);
m_sym_TA_iden = zeros(d_N_f,d_N_OFDM,d_N_antA_act);
m_sym_TA_diag = zeros(d_N_f,d_N_OFDM,d_N_antA_act);

d_N_sig = (d_N_FFT + d_N_CP)*d_N_OFDM;
m_sig_TA_ideal = ones(d_N_sig,4)*(1+1i);
m_sig_TA_iden = ones(d_N_sig,4)*(1+1i);
m_sig_TA_diag = ones(d_N_sig,4)*(1+1i);
m_sig_TB = ones(d_N_sig,4)*(1+1i);

m_H_calib_A2B = zeros(1,2, d_N_f);

for d_f = 1:d_N_f
    %** ideal case **
    v_H_A2B_ideal = squeeze(m_H_est_A2B(:,:,d_f));
    v_P_ideal = v_H_A2B_ideal'/norm(v_H_A2B_ideal);
    m_sym_TA_ideal(d_f,:,:) = (v_P_ideal*m_sym_TA(d_f,:)).';
    %** identity matrix **
    v_H_A2B_iden = squeeze(m_H_est_B2A(:,:,d_f)).';
    v_P_iden = v_H_A2B_iden'/norm(v_H_A2B_iden);
    m_sym_TA_iden(d_f,:,:) = (v_P_iden*m_sym_TA(d_f,:)).';
    %** diagonal calibration **
    v_H_A2B_diag = squeeze(m_H_est_B2A(:,:,d_f).')*diag(m_F(:,d_f));
    v_P_diag = v_H_A2B_diag'/norm(v_H_A2B_diag);
    m_sym_TA_diag(d_f,:,:) = (v_P_diag*m_sym_TA(d_f,:)).';
    m_H_calib_A2B(:,:,d_f) = v_H_A2B_diag;
end

%% -------- signal transmission -------- 
m_sig_TA_ideal(:,v_indA) = f_ofdm_mod(m_sym_TA_ideal,d_N_FFT,d_N_CP,d_N_OFDM,v_active_rfA,d_amp)*2;
m_sig_TA_iden(:,v_indA)  = f_ofdm_mod(m_sym_TA_iden,d_N_FFT,d_N_CP,d_N_OFDM,v_active_rfA,d_amp)*2;
m_sig_TA_diag(:,v_indA)  = f_ofdm_mod(m_sym_TA_diag,d_N_FFT,d_N_CP,d_N_OFDM,v_active_rfA,d_amp)*2;

d_N_sig_R = d_N_OFDM*(d_N_FFT+d_N_CP); 
v_P = exp(1i*2*pi*(0:(d_N_sig_R-1))/4).';
m_P = repmat(v_P,1,4);

oarf_send_frame(card,m_sig_TB,d_n_bit);
m_noise_RB_ = oarf_get_frame(-2);
m_noise_RB = m_noise_RB_(1:d_N_sig,:).*m_P;
m_n_sym_RB = f_ofdm_rx(m_noise_RB, d_N_FFT, d_N_CP, d_N_OFDM, v_active_rfB);

oarf_send_frame(card,m_sig_TA_ideal,d_n_bit);
m_sig_RB_ideal_ = oarf_get_frame(-2);
m_sig_RB_ideal = m_sig_RB_ideal_(1:d_N_sig,:).*m_P;
m_sym_RB_ideal = f_ofdm_rx(m_sig_RB_ideal, d_N_FFT, d_N_CP, d_N_OFDM, v_active_rfB);

oarf_send_frame(card,m_sig_TA_iden,d_n_bit);
m_sig_RB_iden_ = oarf_get_frame(-2);
m_sig_RB_iden = m_sig_RB_iden_(1:d_N_sig,:).*m_P;
m_sym_RB_iden = f_ofdm_rx(m_sig_RB_iden, d_N_FFT, d_N_CP, d_N_OFDM, v_active_rfB);

oarf_send_frame(card,m_sig_TA_diag,d_n_bit);
m_sig_RB_diag_ = oarf_get_frame(-2);
m_sig_RB_diag = m_sig_RB_diag_(1:d_N_sig,:).*m_P;
m_sym_RB_diag = f_ofdm_rx(m_sig_RB_diag, d_N_FFT, d_N_CP, d_N_OFDM, v_active_rfB);


%% -------- SNR measurement -------- 
%** noise measurment **
v_P_n = mean(var(squeeze(m_n_sym_RB),0,2));
%** SNR caculation
%v_P_s_ideal = zeros(301,1);
%for d_f=1:d_N_f
%    v_H_A2B_ideal = squeeze(m_H_est_A2B(:,:,d_f));
%    v_P_s_ideal(d_f) = norm(v_H_A2B_ideal)^2;
%end
%keyboard;
v_P_s_ideal = var(squeeze(m_sym_RB_ideal),0,2);
v_P_s_iden = var(squeeze(m_sym_RB_iden),0,2);
v_P_s_diag = var(squeeze(m_sym_RB_diag),0,2);

v_SNR_ideal_ = 10*log10((v_P_s_ideal-v_P_n)./v_P_n);
v_SNR_iden_ = 10*log10((v_P_s_iden-v_P_n)./v_P_n);
v_SNR_diag_ = 10*log10((v_P_s_diag-v_P_n)./v_P_n);

v_SNR_ideal = nan(d_N_f+1,1);
v_SNR_iden = nan(d_N_f+1,1);
v_SNR_diag = nan(d_N_f+1,1) ;

v_SNR_ideal([1:150 152:301]) = v_SNR_ideal_([1:150 152:301]);
v_SNR_iden([1:150 152:301]) = v_SNR_iden_([1:150 152:301]) ;
v_SNR_diag([1:150 152:301]) = v_SNR_diag_([1:150 152:301]) ;

%save('-v7','result/bf_gain_4x1_t3.mat','v_SNR_ideal','v_SNR_iden','v_SNR_diag','v_SNR_full');
%% -------- plot --------
v_f = d_fc-floor(d_N_f/2)*d_delta_f:d_delta_f:d_fc+ceil(d_N_f/2)*d_delta_f; 
figure(6)
hold on
plot(v_f,v_SNR_ideal,'k-o')
plot(v_f,v_SNR_iden,'g-')
plot(v_f,v_SNR_diag,'r-*')
hold off
%ylim([30 40])

%%------------- Calibration ---------------
m_F_test = zeros(2, d_N_f);
for d_f = 1:d_N_f
  m_F_test(:,d_f) = (squeeze(m_H_est_A2B(:,:,d_f)).')./squeeze(m_H_est_B2A(:,:,d_f));
end

%figure(12)
%subplot(2,1,1)
%hold on;
%for d_f=1:d_N_f
%  plot(m_F_test(1,d_f),'bo')
%  plot(m_F_test(2,d_f),'ro')
%end
%hold off;
%title('Diagonal F');
%axis([-2 2 -2 2])
%grid on
%
%subplot(2,1,2)
%hold on;
%for d_f=1:d_N_f
%  plot(m_F(1,d_f),'bo')
%  plot(m_F(2,d_f),'ro')
%end
%hold off;
%title('Diagonal F');
%axis([-2 2 -2 2])
%grid on

figure(13)
subplot(2,2,1)
plot(20*log10(abs(squeeze(m_H_est_A2B).')),'-');
ylim([0 100])
subplot(2,2,2)
plot(20*log10(abs(squeeze(m_H_calib_A2B).')),'-');
ylim([0 100])
subplot(2,2,3)
plot(angle(squeeze(m_H_est_A2B).'),'-');
subplot(2,2,4)
plot(angle(squeeze(m_H_calib_A2B).'),'-');

