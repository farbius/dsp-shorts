#!/usr/bin/env python
"""
DFT Signal to Noise Ratio 
"""
__author__ = "Aleksei Rostov"
__contact__ = "aleksei.rostov@protonmail.com"
__date__ = "2023/04/10"

import numpy as np
import matplotlib.pyplot as plt


N       = 256      # number of samples
nF0     = 29        # start freq bin

tn_ax   = np.linspace(0,N-1,N)/N; # axis in time domain


def main():

    # Signal with unity power
    x       =  np.exp( 2*1j*np.pi*nF0*tn_ax)
    # Noise with unity power
    n       = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2)
    
    x_pow   = 20*np.log10(np.sum(np.abs(x))/N)
    print("<< Input signal average power is {:3.2f} dB".format(x_pow))
    # noise variance or power
    n_pow   = 10*np.log10(np.var(n))
    print("<< Input noise average power is {:3.2f} dB".format(n_pow))
    
    xF      = np.fft.fft(x)
    nF      = np.fft.fft(n)
    
    n_pow   = 10*np.log10(np.var(nF)/N)
    print("<< Output noise average power is {:3.2f} dB".format(n_pow))
    
    plt.figure(figsize=(15,10))
    
    plt.subplot(2, 1, 1)
    plt.plot(np.real(x), '.-b', label="real(x)")
    plt.plot(np.real(n), '.-r', label="real(n)")
    plt.plot(np.abs(x), '.-g', label="abs(x)")
    plt.ylabel("Modulus")
    plt.xlabel("Time bins")
    plt.title("Signal and additive Noise in time domain")
    plt.legend(loc='upper right')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(xF), '.-b', label="abs(xF)")
    plt.plot(np.abs(nF), '.-r', label="abs(nF)")
    plt.ylabel("Modulus")
    plt.xlabel("Frequency bins")
    plt.title("Signal and additive Noise in freq domain")
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    
    plt.figure(figsize=(15,10))
    plt.plot(20*np.log10(np.abs(xF + nF)/N), '.-b')
    plt.axhline(y=-10*np.log10(N), color='r', linestyle='--', label="Noise floor")
    plt.title("Normalized DFT output, gain is 10*log10(" + "{:4d}".format(N) + ")={:3.2f} dB".format(10*np.log10(N)))
    plt.legend(loc='upper right')
    plt.xlabel('Frequency bins')
    plt.ylabel('Instantaneous Power, dB')
    plt.grid()   
    
    plt.show()

    
    
    
    

if __name__ == "__main__":
    main()