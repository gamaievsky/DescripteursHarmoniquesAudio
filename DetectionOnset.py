from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy import signal
import math
#from scipy import signal

import librosa
import librosa.display

import  params

WINDOW = params.WINDOW
NFFT = int(params.NFFT)
STEP = int(params.STEP)
ALPHA = params.ALPHA
BETA = params.BETA
H = params.H
triFreq = params.triFreq

title = 'Palestrina'
#Palestrina, Cadence4VMaj
y, sr = librosa.load('/Users/manuel/Dropbox (TMG)/Thèse/code/DescripteursHarmoniquesAudio/'+title+'.wav', duration = 6)
Notemin = 'D3'
Notemax = 'D8'

def detectionOnsets(y):
    S = librosa.stft(y, n_fft = NFFT,hop_length= STEP, window=WINDOW)
    Ampl = np.abs(S)
    Phas = np.angle(S)
    Nf = len(Ampl)
    N = len(Ampl[0])
    Ampl_predict = np.zeros((Nf,N))
    Phas_predict = np.zeros((Nf,N))
    Erreur = np.zeros((Nf,N))
    Dev = np.zeros(N)

    if triFreq:
        freqs = librosa.fft_frequencies(sr=sr,n_fft=NFFT)
        fmin = librosa.note_to_hz(Notemin)
        fmax = librosa.note_to_hz(Notemax)
        imin = 0
        while freqs[imin]<fmin:
            imin = imin+1
        imax = imin
        while freqs[imax]<fmax:
            imax = imax+1



    for j in range(2,N):
        for i in range(Nf):
            Ampl_predict[i,j] = Ampl[i,j-1]
            Phas_predict[i,j] = 2*Phas[i,j-1]-Phas[i,j-2]
            #Erreur[i,j] = (Ampl[i,j]**2 + Ampl_predict[i,j]**2 - 2*Ampl_predict[i,j]*Ampl[i,j]*math.cos(Phas[i,j]-Phas_predict[i,j]))**(1/2)
            Erreur[i,j] = math.cos(Phas[i,j]-Phas_predict[i,j])
        if triFreq:
            Dev[j] = sum(Erreur[imin:(imax+1),j])
        else: Dev[j] = sum(Erreur[:,j])

    # Fonction de seuil
    # Ajout de zéros en queue et en tête
    l = []
    Seuil = []
    Onsets = []
    for k  in range(int(H/2)):
        l.append(0)
    for val in Dev:
        l.append(val)
    for k  in range(int(H/2)):
        l.append(0)
    #Calcul de la médiane
    for i in range(N):
        Seuil.append(ALPHA + BETA*stat.median(l[i:i+H]))
        if Dev[i] > Seuil[i]:
            Onsets.append(i)


    times = librosa.frames_to_time(np.arange(N), sr=sr, hop_length=STEP, n_fft=NFFT)
    plt.figure()

    ax1 = plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(Ampl, ref=np.max), sr=sr, hop_length=STEP, x_axis='time', y_axis='log')
    plt.title('Power spectrogram')


    plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(times, Dev, label='Deviation')
    plt.plot(times, Seuil, color='g', label='Seuil')
    plt.vlines(times[Onsets], 0, Dev.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    plt.axis('tight')
    plt.legend(frameon=True, framealpha=0.75)
    plt.show()



detectionOnsets(y)
