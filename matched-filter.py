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

dev     = 20e6      # Hz,  chirp frequency deviation
T0      = N/4*dt    # s,   Pulse length
Sr      = dev/T0/2  # Hz/s chirp rate
f0      = 0e6       # Hz,  chirp start frequency

qin_dB  = 0         # dB,  input SNR

kb      = 1.38e-23  # W*s/K Boltzman constant
Tsh     = 300       # K,    reciever temperatur
kn      = 4         #       noise factor

# Thermal noise average power (Noise floor)
Pn_db   = 10*np.log10(kb*Tsh*kn*Fs)

# Signal average power
Ps_dB   = qin_dB + Pn_db

t_ax    = np.linspace(0,N,N)*dt;# time axis
f_ax    = np.linspace(0, Fs, N);      # freq axis
Ns      = np.round(T0/dt)                   # length of chirp pulse in samples

MF_gain = 10*np.log10(Ns)
print("<< MF gain is {:4.4f} dB".format(MF_gain))


# print("<< T0*dev is {:4.4f} dB".format(10*np.log10(T0*dev)))


def main():
    # Chirp Pulse
    x       = np.exp(2*1j*np.pi*(f0+Sr*t_ax)*t_ax)*(t_ax<T0)
    n       = np.random.randn(N)+1j*np.random.randn(N)
    n      /= np.sqrt(2)
    # Impulse response
    h       = np.exp(2*1j*np.pi*((dev+f0)-Sr*t_ax)*t_ax)*(t_ax<T0)

    x_pow   = 10*np.log10(np.sum(np.abs(x)**2))
    print("<< Input signal average power is {:3.2f} dB".format(x_pow))
    
    plt.figure(figsize=(15,10))
    plt.plot(np.abs(n))
    plt.grid()
    n_pow   = 10*np.log10(np.sum(np.abs(n)**2)/N)
    print("<< Input noise average power is {:3.2f} dB".format(n_pow))
    # print("<< Input SNR is {:4.4f} dB".format(x_pow - n_pow))
    

    # FFT input signal and impulse response
    xF      = np.fft.fft(x)
    nF      = np.fft.fft(n)
    hF      = np.fft.fft(h)
    
    # multiplication in frequency domain
    multF_xh= xF*hF
    multF_nh= nF*hF
    # IFFT
    mult_x  = (np.fft.ifft(multF_xh))/N # np.fft.fftshift
    mult_n  = (np.fft.ifft(multF_nh))/N # np.fft.fftshift
    y       = mult_x+mult_n
    
    x_out_p = 10*np.log10(np.max(np.abs(y)**2))
    print("<< Output signal max power is {:3.2f} dB".format(x_out_p))
    
    n_out_p= 10*np.log10(np.sum(np.abs(mult_n)**2)/N)
    print("<< Output noise average power is {:3.2f} dB".format(n_out_p))
    print("<< Output SNR is {:4.4f} dB".format(x_out_p-n_out_p))

    # Plot
    plt.figure(figsize=(15,10))
    x_ticks = np.arange(min(t_ax/1e-6), max(t_ax/1e-6)+1e-6, 1.0)

    plt.subplot(2, 2, 1)
    plt.plot(t_ax/1e-6, np.real(n), '.-r', label="$P_{ave}$")# + " noise  is {:3.2f} dB".format(n_pow))
    plt.plot(t_ax/1e-6, np.real(x), '.-b', label="$P_{ave}$")# + " signal is {:3.2f} dB".format(x_pow))
    plt.xlabel('t, usec')
    plt.ylabel('Amplitude, V')
    plt.title("MF Input: time domain")
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
    plt.plot(f_ax/1e6, np.abs(hF), '.-b', label='modulus')
    plt.xlabel('f, MHz')
    plt.ylabel('Modulus')
    plt.title("Impulse response: frequency domain ")
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(t_ax/1e-6, 20*np.log10(np.abs(y)), '.-b')
    plt.xlabel('t, usec')
    plt.ylabel('Instantaneous Power, dB')
    plt.axhline(y=x_out_p, color='m', linestyle='--', label="$P_{max}$ signal " + "{:3.2f} dB".format(x_out_p))
    plt.axhline(y=n_out_p, color='g', linestyle='--', label="$P_{ave}$ noise " + "{:3.2f} dB".format(n_out_p))
    plt.title("MF Output: Gain {:3.2f} dB".format(x_out_p-n_out_p))
    plt.legend(loc='upper right')
    plt.xticks(x_ticks)
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()