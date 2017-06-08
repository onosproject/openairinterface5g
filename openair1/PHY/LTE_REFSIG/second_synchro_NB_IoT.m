function [ BOF ] = Fc_second_synchro( new_observation, f_NPSS_symbol , L_frame, L_symbol, FFT_size, L_CP , N_zeros )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%     new_obs_reshape = reshape(new_observation(1:L_frame), L_symbol, []); 
    new_obs_reshape = reshape(new_observation, L_symbol, []); 
    t_new_obs_CP_remov = new_obs_reshape(L_CP+1:end,:); 
    f_new_symbols = fft(t_new_obs_CP_remov,FFT_size); 
    
    for n = 1 : length(f_new_symbols(1,:))
        corr(:,n) = xcorr(f_new_symbols(N_zeros+1:N_zeros+12,n),f_NPSS_symbol); 
        mean = sum(abs(corr(:,n))); 
        mm(n) = sum((abs(corr(:,n))-mean).^2);
    end
    
    for k = 1 : length(mm) - 13
       min_var(k) = sum(mm(k:k+13));  
    end
    [~,BOF] = min(min_var);
end

