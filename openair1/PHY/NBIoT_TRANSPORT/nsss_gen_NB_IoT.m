clear all
% nsss_gen / matlab
% Copyright 2016 b<>com. All rights reserved.

% description: generation of NSSS subframe
% Reference: 3GPP TS36.211 release 13
% author: Vincent Savaux, b<>com, Rennes, France
% email: vincent.savaux@b-com.com

% Input : \
% Output : matrix NSSS_frame
% Parameters
% frame_number = 100;
% cellID = 200;
% % % Mapping results to estimated u-3


SNR_start = -10; 
SNR_end = 2; 
vec_SNR = SNR_start : 2 : SNR_end; 
N_loop = 40; 
Proba_fail = zeros(1,length(vec_SNR)); 

mat_bn = zeros(4,128); % mat_bn contains the 4 possible Hadamard sequences defined in the standard
mat_bn(1,:) = ones(1,128);
mat_bn(2,:) = [1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  1  ...
    -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  ...
    1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  1  -1  1  1  -1  1  -1  ...
    -1  1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  1  -1  1  1  -1  -1  ...
    1  1  -1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  ...
    -1  1  1  -1  1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  ...
    1  -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1];
mat_bn(3,:) = [1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  1  ... 
    -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  -1  1  1  ...
    -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  1  -1  1  ...
    1  -1  -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  -1  ...
    1  1  -1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  ...
    -1  1  1  -1  -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  ...
    -1  1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  1];
mat_bn(4,:) = [1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  1  ...
    -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  -1  1  1  ...
    -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  1  -1  1  ...
    1  -1  -1  1  1  -1  1  -1  -1  1  -1  1  1  -1  1  -1  -1  1  1  ...
    -1  -1  1  -1  1  1  -1  1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  ...
    1  -1  -1  1  1  -1  -1  1  -1  1  1  -1  -1  1  1  -1  1  -1  -1  ...
    1  -1  1  1  -1  1  -1  -1  1  1  -1  -1  1  -1  1  1  -1];
mat_bn = [mat_bn,mat_bn(:,1:4)]; % see the definition of m in stadard

mat_theta_f = zeros(4,132); % mat_bn contains the 4 possible phase sequences defined in the standard
mat_theta_f(1,:) = ones(1,132); 
mat_theta_f(2,:) = repmat([1,-j,-1,j],1,33);  
mat_theta_f(3,:) = repmat([1,-1],1,66); 
mat_theta_f(4,:) = repmat([1,j,-1,-j],1,33); 

mat_16_theta = round(kron(mat_theta_f,ones(4,1))); % mat_bn contains the 4x4=16 possible pseudo-random sequences
mat_16_bn = repmat(mat_bn,4,1); 
mat_16 = mat_16_theta.*mat_16_bn;
corresponding_values = zeros(16,2); % first column for q, second for theta_f
corresponding_values(:,1) = repmat([0;1;2;3],4,1); % mapping column to q 
corresponding_values(:,2) = kron([0;1;2;3],ones(4,1)); % mapping column to theta_f

for k = 1 : length(vec_SNR) % loop on the SNR
    N_fail = 0; 
for loop = 1 : N_loop    
SNR = vec_SNR(k); 
frame_number = 2*randi([0,3],1);
cellID = randi([0,503],1);    
    
% function NSSS_subframe = nsss_gen(frame_number,cellID)
theta_f = 33/132*mod(frame_number/2,4); % as defined in stadard
u = mod(cellID,126) + 3; % root of ZC sequence, defined in standard
q = floor(cellID/126);
size_RB = 12; % number of sub-carrier per RB
N_ZC = 131; 
L_sub_frame = 14; % number of OFDM symbols per subframe
j = 1i; 
vec_n = 0:N_ZC; 
vec_n1 = mod(vec_n,131); 
vec_bq = mat_bn(q+1,:); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creation of the signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ZC sequence in frequency domain

ZC_sequence = exp(-j*pi*u*vec_n1.*(vec_n1+1)/N_ZC); 
had_sequence = exp(-j*2*pi*theta_f*vec_n);
vec_bq_had = vec_bq.*had_sequence; 
P_noise = 10^(-SNR/10); % SNR in dB to noise power
noise = sqrt(P_noise/2)*randn(1,132)+sqrt(P_noise/2)*j*randn(1,132); 
vec_d = vec_bq.*had_sequence.*ZC_sequence + noise; 
mat_NSSS = flipud(reshape(vec_d,size_RB,L_sub_frame-3));
NSSS_subframe = [zeros(size_RB,3),mat_NSSS]; 

% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exhaustive cell ID research
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

sequence_r = repmat(vec_d,16,1).*conj(mat_16); % this remove the phase component
vec_u = 3 : 128; 
mat_u = repmat(vec_u.',1,length(vec_n1));  
mat_n1 = repmat(vec_n1,126,1); 
sequence_ZC = exp(-j*pi*mat_u.*mat_n1.*(mat_n1+1)/N_ZC); 

matrix_max_correl = zeros(126,16); % this will be filled by the maximum of correlation value 
for s_ = 1 : 16 
    seq_ref = sequence_r(s_,:); 
    for u_ = 1 : 126  
       correl = xcorr(seq_ref,sequence_ZC(u_,:)); 
       [val_max,ind_max] = max(abs(correl)); 
       matrix_max_correl(u_,s_) = val_max; 
    end
end

max_correl = max(max(matrix_max_correl));  % get the max of all correlation values 
index_max = find(matrix_max_correl==max_correl); 
estim_u_ = mod(index_max,126)-1;
index_column = (index_max-mod(index_max,126))/126+1; 
estim_q_ = corresponding_values(index_column); 
estim_cell_ID = q*126 + estim_u_; 

if cellID ~= estim_cell_ID
    N_fail = N_fail + 1; 
end

end
Proba_fail(k) = N_fail/N_loop; 
end

plot(vec_SNR,Proba_fail)
