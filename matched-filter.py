#!/usr/bin/env python
"""
The Matched Filter implemenation in frequency domain for Primary Radar Signal processing 
"""
__author__ = "Aleksei Rostov"
__contact__ = "aleksei.rostov@protonmail.com"
__date__ = "2023/03/23"

import numpy as np
import matplotlib.pyplot as plt


N       = 1024      # number of samples
nF1     = 0         # start freq bin
nF2     = 200       # stop  freq bin
Nb      = nF2 - nF1 # spect width
Nt      = N/4       # pulse width
Sn      = Nb/Nt/2*N # chirp rate

tn_ax   = np.linspace(0,1,N); # axis in time domain


def main():

    # Chirp Pulse
    x       =  np.exp(2*1j*np.pi*(nF1+Sn*tn_ax)*tn_ax)*(tn_ax<Nt/N)
    n       = (np.random.normal(0, 1, size=N)+1j*np.random.normal(0, 1, size=N))/np.sqrt(2) 
    # Impulse response
    h       =  np.exp(2*1j*np.pi*((Nb-nF1)-Sn*tn_ax)*tn_ax)*(tn_ax<Nt/N)

    x_pow   = 20*np.log10(np.sum(np.abs(x))/Nt)
    print("<< Input signal average power is {:3.2f} dB".format(x_pow))
    
    n_pow   = 20*np.log10(np.sum(np.abs(n))/N)
    print("<< Input noise average power is {:3.2f} dB".format(n_pow))
    print("<< Input noise average power is {:3.2f} dB".format(np.abs(np.mean(n))))
    
    
    # FFT input signal and impulse response
    xF      = np.fft.fft(x)
    nF      = np.fft.fft(n)
    hF      = np.fft.fft(h)
    
    # multiplication in frequency domain
    multF_xh= xF*hF
    multF_nh= nF*hF
    
    # IFFT
    mult_x  = np.fft.ifft(multF_xh)
    mult_n  = np.fft.ifft(multF_nh)
    y       = mult_x+mult_n
    
    mf_figure(x, h, xF, hF, multF_xh, mult_x, "signal")
    mf_figure(n, h, nF, hF, multF_nh, mult_n, "noise")
    
    n_out_p = 20*np.log10(np.sum(np.abs(mult_n))/N/Nt)
    x_out_p = 20*np.log10(np.max(np.abs(y))/Nt)
    
    plt.figure(figsize=(15,10))
    plt.plot(20*np.log10(np.abs(y)/Nt), '.-r')
    plt.title("Matched Filter Normalized Output: SNR Gain is {:3.2f} dB".format(x_out_p - n_out_p))
    plt.xlabel('Time bins')
    plt.ylabel('Instantaneous Power, dB')
    plt.axhline(y=n_out_p, color='b', linestyle='--', label="$P_{ave}$ noise " + "{:3.2f} dB".format(n_out_p))
    plt.axhline(y=x_out_p, color='g', linestyle='--', label="$P_{max}$ signal " + "{:3.2f} dB".format(x_out_p))
    plt.legend(loc='upper right')
    plt.grid()   
    
    plt.show()
   
    
def mf_figure(xT, hT, xF, hF, xFhF, yT, label="signal"):
    plt.figure(figsize=(15,10))
    plt.suptitle("MF input is a " + label, fontsize=16)
    plt.subplot(2, 2, 1)
    plt.plot(np.abs(xT), '.-g', label="abs(x_in)")
    plt.plot(np.real(xT), '.-b', label="x_in")
    plt.plot(np.real(hT), 'x-r', label="h_mf")
    plt.xlabel("Time bins")
    plt.ylabel("real()")
    plt.legend(loc='upper right')
    plt.title("Time domain")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(np.real(xF), '.-b', label="real(FFT(x_in))")
    plt.plot(np.real(hF), 'x-r', label="real(FFT(h_mf))")
    plt.legend(loc='upper right')
    plt.xlabel("Frequency bins")
    plt.ylabel("real()")
    plt.title("Frequency domain")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(np.real(xFhF), '.-b', label="real(FFT(x_in)*FFT(h_mf))")
    plt.legend(loc='upper right')
    plt.xlabel("Frequency bins")
    plt.ylabel("real()")
    plt.title("Frequency domain")
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(np.abs(yT), '.-b', label="abs(IFFT(FFT(x_in)*FFT(h_mf)))")
    plt.legend(loc='upper right')
    plt.ylabel("abs()")
    plt.xlabel("Time bins")
    plt.title("Time domain")
    plt.tight_layout()
    plt.grid()

   

if __name__ == "__main__":
    main()