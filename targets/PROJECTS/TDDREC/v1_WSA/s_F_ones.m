d_N_antM = 3;
d_N_f = 300;
m_F_norm = ones(d_N_antM, d_N_f)*(1+0*1i);

m_F_Q15 = zeros(d_N_antM, d_N_f*2); 
m_F_Q15(:,1:2:end-1) = floor(real(m_F_norm)*(2^15))-1;
m_F_Q15(:,2:2:end) = floor(imag(m_F_norm)*(2^15));
%%save('-ascii','calibF.mtx','m_F2_diag_Q14');
dlmwrite('result/calibF_iden.mtx', m_F_Q15,' ');
