http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas
Kaldi
HTK
RastaMart
ETS Speech
https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab?focused=5199998&tab=function&s_tid=gn_loc_drop

Input: Waveform in time domain

1) Divide the signal into windows.
Time window: 20ms, Frame shift: 10ms

2) Preemphasis: x[t] - alpha * x[t - 1]. This process boosts up the amplitude of the higher frequency coefficients. Typically, in human audio, the higher frequencies have lower amplitude.

3) Smoothen the signal at boundaries, e.x. hamming window, rectangular window

4) Let's say sampling rate is 16kHz, and you have 20ms of audio, which means 320 samples,
you do a 512-point FFT (nearest power of 2)

Nyquist theorem: You won't be able to calculate the DFT for the frequency >= 0.5* sampling_rate
DFT
0th sample - DC
256th sample - 8kHZ
257 or more > 8KHZ

Therefore, you throw away the DFT samples from 257-512.

5) Pass the DFT samples through Mel FilterBank. (multiply the amplitude ateach freq and then sum) If there are 10 Mel Filters, you will have 10 coefficients. These are called MFSCs or log Mel coefficients.

6) Take inverse DFT of MFSCs, which is called MFCCs

Check Step Liftering

