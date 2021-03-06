# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:48:15 2017

@author: pawel
"""

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
#import matplotlib.pyplot as plt

class voice(object):

    def __init__(self,name):
        self.sig = self.read(name)
        self.smallsiglist = self.ctframes(self.sig)
        self.powerframes = self.fourier1()
        self.fb = self.fbank1()
        self.fbl = self.fbanks()
        self.mfcc = self.mfcc_w_koncu()

    def chunks(self,l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def read(self,filename):
        self.fs, signal = scipy.io.wavfile.read(filename)
        signals = list(self.chunks(signal,int(10*self.fs)))
        for sign in signals:
            sign = self.enhance(sign)
            # print (np.array(np.array(sign).std()).mean())
        return signals

    def enhance(self,signal):
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        emphasized_signal /= max(abs(emphasized_signal))
        return emphasized_signal

    def ctframes(self,signal,frame_size = 0.025,frame_stride = 0.01):
        frame_length, frame_step = frame_size * self.fs, frame_stride * self.fs  # Convert from seconds to samples

        emphasized_signals = self.sig
        frames_list = []
        for emphasized_signal in emphasized_signals:
            signal_length = len(emphasized_signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
            pad_signal_length = num_frames * frame_step + frame_length
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
            indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
            frames = pad_signal[indices.astype(np.int32, copy=False)]
            # obwiednia
            frames *= np.hamming(frame_length)
            frames_list.append(frames)
        return frames_list

    def fourier1(self):
        powf = []
        for frames in self.smallsiglist:
            self.NFFT = 512
            mag_frames = np.absolute(np.fft.rfft(frames, self.NFFT))  # Magnitude of the FFT
            pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))  # Power Spectrum
            powf.append(pow_frames)
        return powf

    def fbank1(self):
        nfilt = 40
        self.low_freq_mel = 0
        self.high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(self.low_freq_mel, self.high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((self.NFFT + 1) * hz_points / self.fs)

        fbank = np.zeros((nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        return fbank

    def fbanks(self):
        fbanklist = []
        for powframe in self.powerframes:
            filter_banks = np.dot(powframe, self.fb.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
            filter_banks = 20 * np.log10(filter_banks)  # dB
            fbanklist.append(filter_banks)
        return fbanklist

    def mfcc_w_koncu(self):
        self.num_ceps = 12 #wymiar wektora2
        mfcclist = []
        for filter_banks in self.fbl:
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (self.num_ceps + 1)] # Keep 2-13

            #filtrowanie, podobno czasami pomaga
            (nframes, ncoeff) = mfcc.shape
            n = np.arange(ncoeff)

            cep_lifter = 22

            lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
            mfcc *= lift  #*

            mfcc -= (np.mean(mfcc, axis=0) + 1e-8) #normalizacja
            mfcclist.append(mfcc)
        return mfcclist


dzwk = voice('krzysiu.wav')
dzwp = voice('pawel.wav')
dzwj = voice('justynka.wav')
dzwm = voice('madzia.wav')
