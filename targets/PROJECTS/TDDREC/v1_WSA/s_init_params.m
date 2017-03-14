clear all
close all
addpath([getenv('OPENAIR_TARGETS') '/ARCH/EXMIMO/USERSPACE/OCTAVE']);
addpath([getenv('OPENAIR_DIR') '/cmake_targets/lte_build_oai/build']);

%% -------- ExpressMIMO2 configuration --------
limeparms;
card = 0;

v_active_rfA = [1 1 0 0];
v_active_rfB = [0 0 1 0];
active_rf = v_active_rfA | v_active_rfB;
if(v_active_rfA*v_active_rfB'~=0) 
    error('The A and B transceive chains must be orthogonal./n') 
end

fc = 2580000000; %1907600000; %1912600000;  %fc = 859.5e6; 
fs = 7.68e6; 
freq_tx = fc*active_rf; 
freq_rx = (fc+fs/4)*active_rf;

tdd_config = DUPLEXMODE_FDD+TXRXSWITCH_LSB;  %we need the LSB switching for the woduplex script, otherwise we don't receive anything
rx_gain = 5*active_rf;
tx_gain = [10 0 5 0];%5*active_rf;
%rx_gain = 20*active_rf;
%tx_gain = 20*active_rf;
syncmode = SYNCMODE_FREE;
eNB_flag = 0;

%rf_mode=(RXEN+TXEN+TXLPFNORM+TXLPFEN+TXLPF25+RXLPFNORM+RXLPFEN+RXLPF25+LNA1ON+LNAByp+RFBBLNA1) * active_rf;
%rf_mode=(TXLPFNORM+TXLPFEN+TXLPF25+RXLPFNORM+RXLPFEN+RXLPF25+LNA1ON+LNAMax+RFBBNORM) * active_rf;
% we have to enable both DMA transfers so that the switching signal in the LSB of the TX buffer gets set
rf_mode = (TXLPFNORM+TXLPFEN+TXLPF25+RXLPFNORM+RXLPFEN+RXLPF5+LNA1ON+LNAMax+RFBBNORM+DMAMODE_TX+TXEN+DMAMODE_RX+RXEN) * active_rf;
rf_rxdc = rf_rxdc*active_rf;                        %???
rf_vcocal = rf_vcocal_19G*active_rf;
rf_local = [8254744   8255063   8257340   8257340]; %eNB2tx 1.9GHz

rffe_rxg_low = 31*active_rf;
rffe_rxg_final = 63*active_rf;
rffe_band = B19G_TDD*active_rf;

autocal_mode = active_rf;
resampling_factor = [2 2 2 2];
oarf_stop(card);
sleep(0.1);
oarf_config_exmimo(0,freq_rx,freq_tx,tdd_config,syncmode,rx_gain,tx_gain,eNB_flag,rf_mode,rf_rxdc,rf_local,rf_vcocal,rffe_rxg_low,rffe_rxg_final,rffe_band,autocal_mode,resampling_factor);

d_n_bit = 16;


