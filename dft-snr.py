#!/usr/bin/env python
"""
DFT Signal to Noise Ratio 
"""
__author__ = "Aleksei Rostov"
__contact__ = "aleksei.rostov@protonmail.com"
__date__ = "2023/03/25"

import numpy as np
import matplotlib.pyplot as plt


N       = 512      # number of samples
nF0     = 16        # start freq bin

nt_ax   = np.linspace(0,1,N); # axis in time domain


def main():

    # Signal
    x       =  np.exp( 2*1j*np.pi*nF0*nt_ax)
    # Reference for nF0 bin DFT
    h       =  np.exp(-2*1j*np.pi*nF0*nt_ax)
    # Noise
    n       = (np.random.normal(0, 1, size=N)+1j*np.random.normal(0, 1, size=N))
    
    xF_nF0  = np.sum(x*h)
    print("<< Signal Modulus on nF0 bin is {:4.1f}".format(np.abs(xF_nF0)))
    nF_nF0  = np.sum(n*h)
    print("<< Noise  Modulus on nF0 bin is {:4.1f}".format(np.abs(nF_nF0)))
    
    xF      = np.fft.fft(x)
    nF      = np.fft.fft(n)
    
    plt.figure()
    
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(xF), '.-b')
    plt.plot(np.abs(nF), '.-r')
    plt.ylabel("Modulus")
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(20*np.log10(np.abs(xF/N)), '.-b')
    plt.plot(20*np.log10(np.abs(nF/N)), '.-r')
    plt.ylabel("Instantaneous Power, dB")
    plt.grid()
    plt.tight_layout()
    
    plt.show()

    
    
    
    

if __name__ == "__main__":
    main()