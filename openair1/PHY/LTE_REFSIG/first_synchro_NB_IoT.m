function [ theta_estim, estim_CFO ] = Fc_first_synchro( observation, L_frame, L_sub_frame, FFT_size, L_symbol, N_subframe_observation, L_CP, SNR, type_first_estim )
% This function performs the estimation of the beginning of symbols as well
% as the estimation of CFO. It allows for a coarse synchronization 

gamma = zeros(1,L_frame);
epsilon = zeros(1,L_frame); 

for n = 1 : 1 : length(gamma)
     
    gamma(n) = sum(observation(n:n+L_CP-1).*conj(observation(n+FFT_size:n+FFT_size+L_CP-1)));
    epsilon(n) = sum(abs(observation(n:n+L_CP-1)).^2 + abs(observation(n+FFT_size:n+FFT_size+L_CP-1)).^2);
end

rho = 10^(SNR/20)/(10^(SNR/20)+1);
theta = 2*abs(gamma)-rho*(epsilon); 

% Estimation of the symbol start and the corresponding CFO

theta_reshape = reshape(theta,L_symbol,N_subframe_observation*L_sub_frame); 
% gamma_reshape = reshape(gamma,L_symbol,N_subframe_observation*L_sub_frame); % useful for estimation of CFO
[~,index_max] = max(theta_reshape); % where theta is max symbol by symbol

switch type_first_estim
    case 1
        %estimation by mean
        theta_estim = sum(index_max)/length(index_max); % estimation by mean
        estim_CFO = -1/(2*pi)*atan(imag(gamma(round(theta_estim)))/real(gamma(round(theta_estim)))); 
    case 2
        %estimation by majority
        counter_index = zeros(1,L_symbol);  
        for k = 1 : 1 : length(index_max)
    
        counter_index(index_max(k)) = counter_index(index_max(k)) + 1; % add the number of index_max
    
        end

        [~,theta_estim] = max(counter_index); % get the max of index max -> theta estim
%         index_index_max = find(index_max == theta_estim); 
%         estim_CFO = -1/(2*pi)*atan(imag(gamma(round(theta_estim)))/real(gamma(round(theta_estim))));
        estim_CFO_vec = -1/(2*pi)*atan(imag(gamma(round(theta_estim:L_symbol:end)))./real(gamma(round(theta_estim:L_symbol:end))));
%         estim_CFO_vec2 = -1/(2*pi)*atan(imag(gamma_reshape(theta_estim,index_max(index_index_max)))./real(gamma_reshape(theta_estim,index_max(index_index_max))));
        estim_CFO = sum(estim_CFO_vec)/length(estim_CFO_vec); 
%         estim_CFO_2 = sum(estim_CFO_vec2)/length(estim_CFO_vec2); 
    otherwise
        print('error: type of estimation not defined')
end

