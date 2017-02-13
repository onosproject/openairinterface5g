dual_tx = 0;

eNBflag = 0;

card    = 0;

limeparms;

%internal loopback test
%rf_mode = [0 0 0 0];
%rf_mode = (               RXEN + TXEN + TXLPFNORM + TXLPFEN + TXLPF5  + RXLPFNORM + RXLPFEN + RXLPF5  + LNA1ON + LNAByp + RFBBLNA1)*[1 1 0 0];
rf_mode  = (RXEN + TXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA2ON + LNAMax + RFBBNORM)*[1 0 0 0];
%rf_mode  = (TXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA2ON + LNAMed + RFBBNORM)*[1 0 0 0];
%rf_mode  = (RXEN + TXLPFNORM + TXLPFEN + TXLPF25 + RXLPFNORM + RXLPFEN + RXLPF25 + LNA2ON + LNAMax + RFBBNORM)*[1 0 0 0];

%% Select DMA directions
%rf_mode = rf_mode + (DMAMODE_RX)*[1 0 0 0 ];
%rf_mode = rf_mode + (DMAMODE_RX)*[1 1 0 0];
%rf_mode = rf_mode + (DMAMODE_TX)*[1 1 0 0];
%rf_mode = rf_mode + (DMAMODE_RX + DMAMODE_TX)*[1 0 0 0];
rf_mode = rf_mode + (DMAMODE_RX + DMAMODE_TX)*[1 0 0 0];
%rf_mode = rf_mode + (DMAMODE_TX)*[1 0 0 0];
%rf_mode = rf_mode + (DMAMODE_RX)*[1 0 0 0];

%freq_rx   = 2662000000*[1 0 0 0];
%freq_rx    = 2685000000*[1 0 0 0];
%freq_rx    = 2684996787*[1 0 0 0];

%freq_rx     = 1910000000*[1 0 0 0];
freq_rx     = 2660000000*[1 0 0 0];
rf_local  = [1156692   8255063   8257340   8257340]; % 1.9 GHz
rf_vcocal = ((0xE)*(2^6)) + (0xE)*[1 1 1 1];         % 1.9 GHz

%freq_tx = freq_rx + 120000000;
%freq_tx = 2000000000*[1 0 0 0];
%freq_tx = 2500000000*[1 0 0 0];
%freq_tx = 2542000000*[1 0 0 0];
freq_tx = 2565000000*[1 0 0 0];
tx_gain = 0*[1 1 1 1];
rx_gain = 50*[1 1 1 1]; 
%rx_gain = 3*[1 1 1 1]; 

%rf_rxdc = rf_rxdc*[1 1 1 1];
rf_rxdc = 32896*[1 1 1 1];

%DUPLEXMODE_FDD=0
%TXRXSWITCH_LSB=0
tdd_config = DUPLEXMODE_FDD + TXRXSWITCH_LSB;

%if (openair0_num_detected_cards==1)
%	p_exmimo_config->framing.multicard_syncmode=SYNCMODE_FREE;
%SYNCMODE_FREE=0
syncmode = SYNCMODE_FREE;

rffe_rxg_low = 63*[1 1 1 1];
rffe_rxg_final = 31*[1 1 1 1];
rffe_band = TVWS_TDD*[1 1 1 1];

%if (openair0_cfg[card].rx_freq[ant] || openair0_cfg[card].tx_freq[ant]) {
%	p_exmimo_config->rf.rf_mode[ant] = RF_MODE_BASE;
%	p_exmimo_config->rf.do_autocal[ant] = 1;//openair0_cfg[card].autocal[ant];
autocal = [1 1 1 1];

%if (openair0_cfg[card].sample_rate==7.68e6)
%	resampling_factor = 2;
%	rx_filter = RXLPF25;
%	tx_filter = TXLPF25;
resampling_factor = [2 2 2 2];

%oarf_config_exmimo(card, freq_rx,freq_tx,0,dual_tx,rx_gain,tx_gain,eNBflag,rf_mode,rf_rxdc,rf_local,rf_vcocal)
oarf_config_exmimo(card, freq_rx, freq_tx, 0, dual_tx, rx_gain, tx_gain, eNBflag, rf_mode, rf_rxdc, rf_local, rf_vcocal, rffe_rxg_low, rffe_rxg_final, rffe_band, autocal, resampling_factor)

% create_testvector
vlen = 307200/pow2(resampling_factor(1));
%v   = ([2:1+vlen; 10002:10001+vlen] - i*[3:2+vlen; 10003:10002+vlen])'; % achtung, wrapped nicht!
%v    = floor( ([640:639+vlen ; 640:639+vlen] / 5) )';


amp = 0;%(pow2(14)-1)/2;
n_bit = 16;

length = 307200/pow2(resampling_factor(1));

s = zeros(length,4);

select = 0; 
chan = 1;

switch(select)

case 0
  s(:,1) = amp * ones(1,length);
  s(:,2) = amp * ones(1,length);
  s(:,3) = amp * ones(1,length);
  s(:,4) = amp * ones(1,length);

case 1
  s(:,1) = floor(amp * (exp(1i*2*pi*(0:((length)-1))/4)));
  s(:,2) = floor(amp * (exp(1i*2*pi*(0:((length)-1))/4)));
  s(:,3) = floor(amp * (exp(1i*2*pi*(0:((length)-1))/4)));
  s(:,4) = floor(amp * (exp(1i*2*pi*(0:((length)-1))/4)));
%	plot(real(s(1:vlen,chan)),'g', "markersize",1); 
%	plot(imag(s(1:vlen,chan)),'y', "markersize",1); 


case 2
	for i = 1:vlen 
		  s(i,1) = (mod(i,pow2(12)-1)-2048)*16;
	endfor
	plot(real(s(1:vlen,chan)),'b', "markersize",1); 

case 3
	for i = 1:vlen 
		  s(i,1) = mod(i,1024);
	endfor
	plot(real(s(1:vlen,chan)),'b', "markersize",1); 

case 6

  nb_rb = 25; %this can be 25, 50, or 100
  num_carriers = 2048/100*nb_rb;
  num_zeros = num_carriers-(12*nb_rb+1);
  prefix_length = num_carriers/4; %this is extended CP
  num_symbols_frame = 120;
  preamble_length = 120;
  
  s(:,1) = OFDM_TX_FRAME(num_carriers,num_zeros,prefix_length,num_symbols_frame,preamble_length);
  s(:,1) = floor(amp*(s(:,1)./max([real(s(:,1)); imag(s(:,1))])));

otherwise 
  error('unknown case')

end %switch

%break

sleep(1)

%s = s*2;

%oarf_send_frame(card, s, 16);

sleep(1)

%len = 1000; off = 0; chan = 2; hold off; plot(real(s(off+1:off+len,chan)), '-o', "markersize",2); hold on; plot(imag(s(off+1:off+len,chan)), 'r-o', "markersize",2)
chan = 1;
s    = oarf_get_frame(card);
hold off;
plot(real(s(1:vlen,chan)),'b', "markersize",1); 
%plot(real(s(:, chan)), 'b', "markersize",1); 
hold on ; 
plot(imag(s(1:vlen,chan)),'r', "markersize",1); 
%plot(imag(s(:,chan)),'r',"markersize",1)

%hold off;
%plot(20*log10(abs(fft(s(:,1)))));
%plot(20*log10(abs(fftshift(fft(s(1:vlen/10,1))))),'r');




gainimb_rx = -0.0020;  phaseimb_rx = -2.38; % ExMIMO1 / lime1, VGAgain2 = 0, 1.9 GHz
gainimb_rx = 0;  phaseimb_rx = -10	; % ExMIMO1 / lime1, VGAgain2 = 0, 1.9 GHz

phaseimb_rx = phaseimb_rx/180*pi; % phaser imb in radians
beta_rx = (1/2)*(1 + (1+ gainimb_rx) * exp(1i*phaseimb_rx));
alpha_rx = (1/2)*(1 - (1+ gainimb_rx) * exp(-1i*phaseimb_rx));
den=abs(beta_rx)^2-abs(alpha_rx)^2;
beta_rx=beta_rx/den;
alpha_rx=alpha_rx/den;

%s2 =  beta_rx.*s + alpha_rx.*conj(s); 
% hold on ; plot(20*log10(abs(fftshift(fft(s2(1:vlen,1))))), 'b'); grid on;


