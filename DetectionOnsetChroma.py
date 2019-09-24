from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from scipy import signal
import math
#from scipy import signal

import librosa
import librosa.display

import params

WINDOW = params.WINDOW
BINS_PER_OCTAVE = params.BINS_PER_OCTAVE
STEP = 512
ALPHA = params.ALPHA
BETA = params.BETA
H = params.H
T = params.T
T_att = params.T_att
plot_onsets = params.plot_onsets
norm_spectre = params.norm_spectre

title = 'Palestrina'
#Palestrina, Cadence4VMaj
y, sr = librosa.load('/Users/manuel/Github/DescripteursHarmoniquesAudio/Exemples/'+title+'.wav')
Notemin = 'D3'
Notemax = 'D9'




def detectionOnsets(y):
    fmin = librosa.note_to_hz(Notemin)
    fmax = librosa.note_to_hz(Notemax)
    #Nmin = int((sr/(fmax*(2**(1/BINS_PER_OCTAVE)-1))))
    #Nmax = int((sr/(fmin*(2**(1/BINS_PER_OCTAVE)-1))))
    n_bins = int((librosa.note_to_midi(Notemax) - librosa.note_to_midi(Notemin))*BINS_PER_OCTAVE/12)
    Chrom = librosa.amplitude_to_db(np.abs(librosa.cqt(y=y, sr=sr, hop_length = STEP, fmin= fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=n_bins)), ref=np.max)
    Nf = len(Chrom)
    N = len(Chrom[0])
    Diff = np.zeros((Nf,N))
    Dev = np.zeros(N)
    for j in range(1,N):
        for i in range(Nf):
            Diff[i,j] = np.abs(Chrom[i,j]-Chrom[i,j-1])
            Dev[j] = sum(Diff[:,j])

    # FONCTION DE SEUIL
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


    times = librosa.frames_to_time(np.arange(N), sr=sr, hop_length=STEP)

    # FONCTION DE TRI SUR LES  ONSETS
    i=0
    while i<(len(Onsets)-1):
        while (i<(len(Onsets)-1)) and (times[Onsets[i+1]]< times[Onsets[i]]+T):
            if Dev[Onsets[i+1]] < Dev[Onsets[i]]: del Onsets[i+1]
            else: del Onsets[i]
        i=i+1


    onset_frames = librosa.util.fix_frames(Onsets, x_min=0, x_max=Chrom.shape[1]-1)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length = STEP)

    #Synchronisation sur les onsets, en enlevant le début et la fin des longues frames
    ChromSync = np.zeros((Nf,len(onset_frames)-1))
    n_att = int(librosa.time_to_frames(T_att, sr=sr, hop_length = STEP))
    for j in range(len(onset_frames)-1):
        for i in range(Nf):
            ChromSync[i,j] = np.mean(Chrom[i][(onset_frames[j]+n_att):(onset_frames[j+1]-n_att)])

    #Normalisation du spectre
#    ChromSync[:,1] = librosa.power_to_db(librosa.db_to_power(ChromSync[:,1]) / np.sum(librosa.db_to_power(ChromSync[:,1])))
    if norm_spectre:
        for j in range(ChromSync.shape[1]):
            ChromSync[:,j] = librosa.power_to_db(librosa.db_to_power(ChromSync[:,j]) / np.sum(librosa.db_to_power(ChromSync[:,j])))




    #Affichage
    if plot_onsets:
        plt.figure(figsize=(13, 7))
        ax1 = plt.subplot(3, 1, 1)
        librosa.display.specshow(Chrom, bins_per_octave=BINS_PER_OCTAVE, fmin=fmin, y_axis='cqt_note', x_axis='time', x_coords=times)
        plt.title('CQT spectrogram')

        plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(times, Dev, label='Deviation')
        plt.plot(times, Seuil, color='g', label='Seuil')
        plt.vlines(times[Onsets], 0, Dev.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
        plt.axis('tight')
        plt.legend(frameon=True, framealpha=0.75)

        ax1 = plt.subplot(3, 1, 3, sharex=ax1)
        librosa.display.specshow(ChromSync, bins_per_octave=BINS_PER_OCTAVE, fmin=fmin, y_axis='cqt_note', x_axis='time',x_coords=onset_times)
        plt.show()

    return onset_times


onset_times = detectionOnsets(y)
