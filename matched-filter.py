#!/usr/bin/env python
"""
The Matched Filter implemenation in frequency domain for Primary Radar Signal processing 
"""
__author__ = "Aleksei Rostov"
__contact__ = "aleksei.rostov@protonmail.com"
__date__ = "2023/03/23"

import numpy as np
import matplotlib.pyplot as plt


N       = 1024       #      number of points
Fs      = 100e6     # Hz,  sample rate
dt      = 1/Fs

dev     = 20e6       # Hz,  chirp frequency deviation
T0      = (N/2-1)*dt      # s,   Pulse length
Sr      = dev/T0/2  # Hz/s chirp rate
f0      = 0e6       # Hz,  chirp start frequency
Ps_dB   = 0         # dB,  signal power
Pn_dB   = 0         # dB,  noise  power

t_ax    = np.linspace(0,N,N)*dt;# time axis
f_ax    = np.linspace(0, Fs, N);      # freq axis
Ns      = np.round(T0/dt)                   # length of chirp pulse in samples

MF_gain = 10*np.log10(Ns)
print("<< MF gain is {:4.4f} dB".format(MF_gain))

def main():
    # delayed pulse signal
    x       = 10**(Ps_dB/20)*np.exp(2*1j*np.pi*(f0+Sr*t_ax)*t_ax)*(t_ax<T0)
    n       = 10**(Pn_dB/20)*(np.random.randn(N)+1j*np.random.randn(N))
    # impulse response
    h       = np.exp(2*1j*np.pi*((dev+f0)-Sr*t_ax)*t_ax)*(t_ax<T0)

    x_pow   = 10*np.log10(np.sum(np.abs(x)**2)/Ns)
    print("<< Input signal average power is {:3.2f} dB".format(x_pow))
    n_pow   = 10*np.log10(np.sum(np.abs(n)**2)/N/2)
    print("<< Input noise average power is {:3.2f} dB".format(n_pow))

    # FFT input signal and impulse response
    xF      = np.fft.fft(x)/N
    nF      = np.fft.fft(n)/N
    hF      = np.fft.fft(h)/N
    
    # multiplication in frequency domain
    multF_xh= xF*hF
    multF_nh= nF*hF
    # IFFT
    mult_x  = (np.fft.ifft(multF_xh))*N
    mult_n  = (np.fft.ifft(multF_nh))*N
    
    mult_x_p= 20*np.log10(np.max(np.abs(mult_x)))
    print("<< Output signal average power is {:3.2f} dB".format(mult_x_p))
    
    mult_n_p= 20*np.log10(np.sum(np.abs(mult_n))/N)
    print("<< Output noise average power is {:3.2f} dB".format(mult_n_p))
    
    y       = mult_x + mult_n
    
    plt.figure()
    plt.plot(20*np.log10(np.abs(mult_x)), '.-r')
    plt.plot(20*np.log10(np.abs(mult_n)), '.-b')
    plt.axhline(y=mult_x_p, color='m', linestyle='--', label="$P_{max}$ signal " + "{:3.2f} dB".format(mult_x_p))
    plt.axhline(y=mult_n_p, color='g', linestyle='--', label="$P_{ave}$ noise " + "{:3.2f} dB".format(mult_n_p))
    plt.legend(loc='upper right')
    plt.xlabel('samples')
    plt.ylabel('Instantaneous Power, dB')
    plt.title("Matched Filter Output: SNR = {:3.2f}  dB".format(mult_x_p - mult_n_p))
    plt.ylim([-60, 20])
    plt.grid()

    # # Plot
    plt.figure(figsize=(15,10))
    x_ticks = np.arange(min(t_ax/1e-6), max(t_ax/1e-6)+1e-6, 1.0)

    plt.subplot(2, 2, 1)
    plt.plot(t_ax/1e-6, np.real(n), '.-r', label="$P_{ave}$" + " noise  is {:3.2f} dB".format(n_pow))
    plt.plot(t_ax/1e-6, np.real(x), '.-b', label="$P_{ave}$" + " signal is {:3.2f} dB".format(x_pow))
    plt.xlabel('t, usec')
    plt.ylabel('Amplitude, V')
    plt.title("Matched Filter Input: time domain")
    plt.legend(loc='upper right')
    plt.xticks(x_ticks)
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t_ax/1e-6, np.real(h), '.-b', label='real')
    plt.xlabel('t, usec')
    plt.ylabel('Amplitude, V')
    plt.title("Impulse response: time domain")
    plt.legend(loc='upper right')
    plt.xticks(x_ticks)
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(f_ax/1e6, 20*np.log10(np.abs(hF)), '.-b', label='modulus')
    plt.xlabel('f, MHz')
    plt.ylabel('Modulus')
    plt.title("Impulse response: frequency domain ")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(t_ax/1e-6, 20*np.log10(np.abs(y)) - MF_gain, '.-b')
    plt.xlabel('t, usec')
    plt.ylabel('Instantaneous Power, dB')
    plt.axhline(y=MF_gain + x_pow, color='r', linestyle='--', label="$P_{ave}$ signal + MF gain")
    plt.axhline(y=n_pow, color='g', linestyle='--', label="$P_{ave}$ noise")
    plt.title("Matched Filter Output: input SNR is {:3.2f} dB, MF gain is {:3.2f} dB".format(x_pow - n_pow, MF_gain))
    plt.legend(loc='upper right')
    plt.xticks(x_ticks)
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()