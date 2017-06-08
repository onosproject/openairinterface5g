clear all 
close all 

% description: test synchro using CP for time-freq. synchro
% and ZC sequence for beginning of the radio frame estimation
% date : 09/03/2017
% author : Vincent Savaux, b<>com, Rennes, France
% email: vincent.savaux@b-com.com

% Parameters

u = 5; % root of ZC sequence 
size_RB = 12; % number of sub-carrier per RB 
N_ZC = size_RB-1; 
L_sub_frame = 14; 
j = 1i; 
CFO = 0.1; % normalized CFO 
N_frames = 4; % at least 3 
N_sub_frame = 10*N_frames; % how many simulated sub_frame you want 
FFT_size = 128; 
N_zeros = (FFT_size-size_RB)/2; % Number of zero subcarriers in upper and lower frequencies 
L_CP = round(4.6875/(66.7)*FFT_size); % Number of samples of the CP 
L_symbol = (FFT_size + L_CP); 
L_frame = (FFT_size + L_CP)*L_sub_frame*10; 
L_signal = (FFT_size + L_CP)*L_sub_frame*N_sub_frame; 
normalized_time = 0 : 1 : L_signal-1; 
SNR_start = 0; % in dB 
SNR_end = 30; % in dB 
N_subframe_observation = 10; % length of observation for syncronization 
N_loop = 1000; % number of runs, for good statistics 
type_first_estim = 2; % 1 -> estimation by mean, 2-> estimation by majority 

matrix_error_theta_1 = zeros(N_loop,SNR_end-SNR_start+1); 
matrix_error_theta_2 = zeros(N_loop,SNR_end-SNR_start+1);  
matrix_error_angle_1 = zeros(N_loop,SNR_end-SNR_start+1); 
matrix_error_angle_2 = zeros(N_loop,SNR_end-SNR_start+1); 
matrix_error_angle_3 = zeros(N_loop,SNR_end-SNR_start+1); 
matrix_error_BOF = zeros(N_loop,SNR_end-SNR_start+1); 

for SNR = SNR_start : 2 : SNR_end
for loop = 1 : N_loop     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creation of the signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ZC sequence in frequency domain

vec_n = 0:N_ZC-1; 
f_ZC_sequence = exp(-j*pi*u*vec_n.*(vec_n+1)/N_ZC); 
f_NPSS_symbol = [f_ZC_sequence.';0]; % one NPSS symbol 
f_NPSS_frame = [zeros(size_RB,3),kron(ones(1,L_sub_frame-3),f_NPSS_symbol)]; 

% OFDM sub_frame in frequency domain -> modulation : QPSK
%random QPSK elements: 
f_OFDM_frames = (2*randi([0,1],size_RB,L_sub_frame*N_sub_frame)-1) + j*(2*randi([0,1],size_RB,L_sub_frame*N_sub_frame)-1); 

%replace the k*6th subframes by f_NPSS_frame:
f_LTE_frames = f_OFDM_frames; 
for k = 0 : N_frames-1
   
    N_index = k*10*L_sub_frame + 85; 
    f_LTE_frames(:,N_index:N_index+13) = f_NPSS_frame; 
    
end

% IFFT: get frames in time domain (Parralel representation)

f_zero_carriers = zeros(N_zeros,L_sub_frame*N_sub_frame); 
f_ups_LTE_frames = [f_zero_carriers;f_LTE_frames;f_zero_carriers]; % add zero carriers up and down the symbols
t_P_LTE_frames = ifft(f_ups_LTE_frames,FFT_size); 

% Add CP (Parralel representation) 

t_P_LTE_frames_CP = [t_P_LTE_frames(end-L_CP+1:end,:);t_P_LTE_frames]; 

% Parralel to series conversion 

t_S_LTE_frames = reshape(t_P_LTE_frames_CP,1,[]); 

% Add a channel frequency offset (CFO)

t_S_received_frames = t_S_LTE_frames.*exp(j*2*pi*CFO*normalized_time/FFT_size); 

% Add noise 

P_signal = sum(abs(t_S_received_frames).^2)/length(t_S_received_frames); 
P_noise = P_signal*10^(-SNR/20); 

init_noise = randn(size(t_S_received_frames));
normalized_noise = init_noise/sqrt(sum(abs(init_noise).^2)/length(init_noise));
noise = sqrt(P_noise)*normalized_noise;

t_S_noisy_frames = t_S_received_frames + noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time and frequency synchronization
% The principle is based on the croos-correlation between the received
% sequence and the transmitted SC sequence. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get an observation, the duration of which is one frame length. The
% beginning of the stored samples is at 1 frame +- 0.5 frame

index_start = L_frame + randi([-L_frame/2,L_frame/2],1);
observation = t_S_noisy_frames(index_start:index_start+1.5*L_frame-1);
f_oversampl_NPSS_symbol = [f_zero_carriers(:,1);f_NPSS_symbol;f_zero_carriers(:,1)];
t_NPPS_unit = ifft(f_oversampl_NPSS_symbol,FFT_size); 
t_NPPS_correl = [t_NPPS_unit(end-L_CP+1:end);t_NPPS_unit]; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation of start of the symbols and the CFO 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[ theta_estim, estim_CFO ] = first_synchro( observation, L_frame, L_sub_frame, FFT_size, L_symbol, N_subframe_observation, L_CP, SNR, type_first_estim ); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time and frequency synchronization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

new_index_start = index_start + theta_estim - 1; 
new_vec_time = new_index_start-1 : 1 : new_index_start+1.5*L_frame-2; 
new_observation = t_S_noisy_frames(new_index_start:new_index_start+1.5*L_frame-1).*exp(-j*2*pi*estim_CFO*new_vec_time/FFT_size); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Second synchronization: beginning of frame (BOF)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

[ BOF ] = second_synchro( new_observation, f_NPSS_symbol , L_frame, L_symbol, FFT_size, L_CP, N_zeros ); 


exact_index = L_symbol - rem(index_start,L_symbol) + 2; 
[error_theta_1,ind_error1] = min([abs(exact_index - theta_estim-L_symbol), abs(exact_index - theta_estim),abs(exact_index - theta_estim+L_symbol)]); 
% [error_theta_2,ind_error2] = min([abs(exact_index - theta_estim_2-137), abs(exact_index - theta_estim_2),abs(exact_index - theta_estim_2+137)]); 

error_CFO_1 = CFO - estim_CFO; 
% error_CFO_2 = CFO - estim_CFO_1; 
% error_CFO_3 = CFO - estim_CFO_2; 

vec_exact_index = 85 : 140 : N_frames*140;  
estim_BOF = ceil((index_start-1)/L_symbol) + BOF ; 
error_BOF = min(abs(estim_BOF-vec_exact_index)); 

matrix_error_theta_1(loop,SNR-SNR_start+1) = error_theta_1; 
% matrix_error_theta_2(loop,SNR-SNR_start+1) = error_theta_2;  
matrix_error_angle_1(loop,SNR-SNR_start+1) = error_CFO_1; 
% matrix_error_angle_2(loop,SNR-SNR_start+1) = error_CFO_2; 
% matrix_error_angle_3(loop,SNR-SNR_start+1) = error_CFO_3;
matrix_error_BOF(loop,SNR-SNR_start+1) = error_BOF;

end
end
plot(SNR_start:2:SNR_end,sqrt(sum(abs(matrix_error_theta_1(:,1:2:SNR_end+1)).^2)/N_loop))
hold
% plot(SNR_start:2:SNR_end,sqrt(sum(abs(matrix_error_theta_2(:,1:2:SNR_end+1)).^2)/N_loop))
figure
plot(SNR_start:2:SNR_end,sqrt(sum(abs(matrix_error_angle_1(:,1:2:SNR_end+1)).^2)/N_loop))
hold
% plot(SNR_start:SNR_end,sum(abs(matrix_error_angle_2(:,1:31)))/N_loop)
% plot(SNR_start:SNR_end,sum(abs(matrix_error_angle_3(:,1:31)))/N_loop)
% plot(SNR_start:2:SNR_end,sqrt(sum(abs(matrix_error_angle_2(:,1:2:SNR_end+1)).^2)/N_loop),'s')
% plot(SNR_start:2:SNR_end,sqrt(sum(abs(matrix_error_angle_3(:,1:2:SNR_end+1)).^2)/N_loop),'d')
figure 
plot(SNR_start:2:SNR_end,sqrt(sum(abs(matrix_error_BOF(:,1:2:SNR_end+1)).^2)/N_loop))
hold
save

