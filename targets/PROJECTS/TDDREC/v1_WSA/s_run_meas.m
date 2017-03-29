%
% SCRIPT ID : s_run_meas
%
% PROJECT NAME : TDD Recoprocity
%
% PURPOSE : full transmission and receive train for TDD reciprocity calibration
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
% Apr-29-2014  X. JIANG       0.1     script creation v0.1
%
%  REFERENCES/NOTES/COMMENTS :
%
% - Based on the script "run_full_duplex" created by Mirsad Cirkic, Florian Kaltenberger.
% 
%**********************************************************************************************

%% ** initialisation **
%% ------------- to change in experiment ------------
%clc
close all
%clear all
% 
%d_M = 4;				% modulation order, e.g. 4 means QPSK
%
%%%** frequency **
%d_N_f = 301; 			% carrier number carrying data
%d_N_FFT = 512;			% total carrier number
%d_N_CP = 128;			% extented cyclic prefix
%%** time **
%d_N_OFDM = 120;			% number of ofdm symbol per frame
%d_N_meas = 10;			% measuement number
%%** antenna **
%d_N_antA = 4;			% antenna number at site a
%d_N_antB = 4; 			% antenna number at site b
%v_active_rfA=[0 0 1 0];		%temp
%v_active_rfB=[1 1 0 0];
%v_indA = find(v_active_rfA);	% active antenna index at site a
%v_indB = find(v_active_rfB);	% active antenna index at site b
%d_amp = 10;
%% ----------------------------------------------------
m_sym_T = zeros(d_N_f,d_N_OFDM,3,d_N_meas);
m_sym_TA = zeros(d_N_f,d_N_OFDM/2,length(v_indA),d_N_meas);
m_sym_TB = zeros(d_N_f,d_N_OFDM/2,length(v_indB),d_N_meas);
m_sym_RA = zeros(d_N_f,d_N_OFDM/2,length(v_indA),d_N_meas);
m_sym_RB = zeros(d_N_f,d_N_OFDM/2,length(v_indB),d_N_meas);

%** simulation**
%m_sig_R = zeros((d_N_FFT+d_N_CP)*d_N_OFDM,4);

for d_n_meas = 1:d_N_meas
    %% -------- tx -------- 
    %** tx of site A **
    [m_sym_T(:,:,:,d_n_meas), m_sig_T] = f_ofdm_tx(d_M, d_N_f, d_N_FFT, d_N_CP, d_N_OFDM, d_N_antA, active_rf, d_amp);
    m_sym_TA(:,:,:,d_n_meas) = m_sym_T(:,1:end/2,1:end-1,d_n_meas);
    m_sym_TB(:,:,:,d_n_meas) = m_sym_T(:,end/2+1:end,end,d_n_meas); 
    %** simulation
    %m_sym_TA(:,d_N_OFDM/4+1:d_N_OFDM/2,1,d_n_meas) = 0;
    %m_sig_T(end/4+1:end/2,v_indA(1)) = 0; 
    %m_sym_TA(:,1:d_N_OFDM/4,2,d_n_meas) = 0;
    %m_sig_T(1:end/4,v_indA(2)) = 0;
 
    %m_sig_T(1:end/2, v_indA) = m_sig_T(1:end/2,v_indA);
    %m_sig_T(end/2+1:end, v_indA) = 0;
    %m_sig_T(end/2+1:end, v_indB) = m_sig_T(end/2+1:end,v_indB);
    %m_sig_T(1:end/2, v_indB) = 0;

    %m_sig_R(end/2+1:end,1) = m_sig_T(end/2+1:end,4);
    %m_sig_R(end/2+1:end,2) = m_sig_T(end/2+1:end,4);
    %m_sig_R(1:end/2,4) = m_sig_T(1:end/2,1)+m_sig_T(1:end/2,2);

    %** prepare the signal ** 
    m_sym_TA(:,d_N_OFDM/4+1:d_N_OFDM/2,v_indA(1),d_n_meas) = 0;
    m_sym_TA(:,1:d_N_OFDM/4,v_indA(2),d_n_meas) = 0;
 
    m_sig_T(1:end/2, v_indA) = m_sig_T(1:end/2,v_indA)*2;
    m_sig_T(end/2+1:end, v_indB) = m_sig_T(end/2+1:end,v_indB)*2;

    m_sig_T(end/2+1:end, v_indA) = 1+1i;
    m_sig_T(1:end/2, v_indB) = 1+1i;
    m_sig_T(end/4+1:end/2,v_indA(1)) = 1+1i; 
    m_sig_T(1:end/4,v_indA(2)) = 1+1i;

    %% -------- channel --------     
    %** Transmission from A to B **
    oarf_send_frame(card,m_sig_T,d_n_bit);
    m_sig_R_ = oarf_get_frame(-2);
    
    d_N_sig_R = d_N_OFDM*(d_N_FFT+d_N_CP); 
    v_P = exp(1i*2*pi*(0:(d_N_sig_R-1))/4).';
    m_sig_R = m_sig_R_(1:d_N_sig_R,:) .* repmat(v_P,1,size(m_sig_R_,2)); 

    m_sig_RA = m_sig_R(end/2+1:end,:);
    m_sig_RB = m_sig_R(1:end/2,:);

    %% -------- rx --------  
    m_sym_RB(:,:,:,d_n_meas) = f_ofdm_rx(m_sig_RB, d_N_FFT, d_N_CP, d_N_OFDM/2, v_active_rfB);
    m_sym_RA(:,:,:,d_n_meas) = f_ofdm_rx(m_sig_RA, d_N_FFT, d_N_CP, d_N_OFDM/2, v_active_rfA);

end

%    keyboard;
%** channel estimation **
m_H_est_A2B = f_ch_est(m_sym_TA, m_sym_RB);           %dimension: d_N_antR x d_N_antT x d_N_f x d_N_meas
m_H_est_B2A = f_ch_est(m_sym_TB, m_sym_RA);

%% -------- plot --------

%** channel estimation in frequency domain **
 m_H_A2B_draw = squeeze(m_H_est_A2B(1,:,:,1)).';
 m_H_B2A_draw = squeeze(m_H_est_B2A(:,1,:,1)).';
 %keyboard 

 figure(1)
 subplot(2,1,1)
 plot(real(m_sig_RA(:,v_indA)),'-');
 title('m_sig_RA')
 subplot(2,1,2)
 plot(real(m_sig_RB(:,v_indB)),'b-');
 hold on
 plot(real(m_sig_RB(end-100:end,v_indB)),'r-')
 title('m_sig_RB')

 figure(2)
 subplot(2,2,1)
 plot(20*log10(abs(m_H_A2B_draw)),'-');
 title('|h| vs. freq (A2B)')
 xlabel('freq')
 ylabel('|h|')
 ylim([0 100])
 
 subplot(2,2,2)
 plot(20*log10(abs(m_H_B2A_draw)),'-');
 title('|h| vs. freq (B2A)')
 xlabel('freq')
 ylabel('|h|')
 ylim([0 100])

 subplot(2,2,3)
 plot(angle(m_H_A2B_draw),'-');
 title('angle(h) vs. freq (A2B)')
 xlabel('freq')
 ylabel('angle(h)')
 
 subplot(2,2,4)
 plot(angle(m_H_B2A_draw),'-');
 title('angle(h) vs. freq (B2A)')
 xlabel('freq')
 ylabel('angle(h)')

 figure(3)
 plot(m_sym_RA(1,:,1,1),'b*')
% hold on     
% plot(m_sym_RA(1,:,1,3),'r*')
% hold on     
% plot(m_sym_RA(1,:,1,5),'g*')
 title('m sym RA 1')

 figure(4)
 plot(m_sym_RA(1,:,2,1),'b*')
% hold on            
% plot(m_sym_RA(1,:,2,3),'r*')
% hold on            
% plot(m_sym_RA(1,:,2,5),'g*')
 title('m sym RA 2')

 figure(5)
 subplot(2,1,1)
 plot(m_sym_RB(1,1:end/2,1,1),'b*')
% hold on
% plot(m_sym_RB(1,1:end/2,1,3),'r*')
% hold on
% plot(m_sym_RB(1,1:end/2,1,5),'g*')
 title('m sym RB ant 2')
 subplot(2,1,2)
 plot(m_sym_RB(1,end/2+1:end,1,1),'b*')
% hold on
% plot(m_sym_RB(1,end/2+1:end,1,3),'r*')
% hold on
% plot(m_sym_RB(1,end/2+1:end,1,5),'g*')
 title('m sym RB ant 2')
