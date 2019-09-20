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
plot_pistes = params.plot_pistes
norm_spectre = params.norm_spectre



class SignalSepare:
    """ Prend en entrée en signal et le signal des pistes audio séparées. """

    def __init__(self, signal, sr, pistes, Notemin  = 'D3', Notemax = 'D9'):
        self.y = signal
        self.pistes = pistes
        self.sr = sr
        self.n_pistes = len(pistes)
        self.Notemin = Notemin
        self.Notemax = Notemax
        self.Nf = 0
        self.N = 0
        self.fmin = 0
        self.fmax = 0
        self.n_bins = 0
        self.n_att = 0
        self.n_frames = 0
        self.onset_times = []
        self.onset_frames = []
        self.Chrom = []
        self.chromSync = []
        self.chromPistesSync = []
        self.chromConc = []
        self.concordance = []

    def detectionOnsets(self):
        self.fmin = librosa.note_to_hz(self.Notemin)
        self.fmax = librosa.note_to_hz(self.Notemax)
        #Nmin = int((sr/(fmax*(2**(1/BINS_PER_OCTAVE)-1))))
        #Nmax = int((sr/(fmin*(2**(1/BINS_PER_OCTAVE)-1))))
        self.n_bins = int((librosa.note_to_midi(self.Notemax) - librosa.note_to_midi(self.Notemin))*BINS_PER_OCTAVE/12)
        self.Chrom = librosa.amplitude_to_db(np.abs(librosa.cqt(y=self.y, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)), ref=np.max)
        self.Nf = len(self.Chrom)
        self.N = len(self.Chrom[0])
        Diff = np.zeros((self.Nf,self.N))
        Dev = np.zeros(self.N)
        for j in range(1,self.N):
            for i in range(self.Nf):
                Diff[i,j] = np.abs(self.Chrom[i,j]-self.Chrom[i,j-1])
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
        for i in range(self.N):
            Seuil.append(ALPHA + BETA*stat.median(l[i:i+H]))
            if Dev[i] > Seuil[i]:
                Onsets.append(i)


        times = librosa.frames_to_time(np.arange(self.N), sr=sr, hop_length=STEP)

        # FONCTION DE TRI SUR LES  ONSETS
        i=0
        while i<(len(Onsets)-1):
            while (i<(len(Onsets)-1)) and (times[Onsets[i+1]]< times[Onsets[i]]+T):
                if Dev[Onsets[i+1]] < Dev[Onsets[i]]: del Onsets[i+1]
                else: del Onsets[i]
            i=i+1


        self.onset_frames = librosa.util.fix_frames(Onsets, x_min=0, x_max=self.Chrom.shape[1]-1)
        self.onset_times = librosa.frames_to_time(self.onset_frames, sr=sr, hop_length = STEP)
        self.n_frames = len(self.onset_frames)-1


        #Synchronisation sur les onsets, en enlevant le début et la fin des longues frames
        self.chromSync = np.zeros((self.Nf,self.n_frames))
        self.n_att = int(librosa.time_to_frames(T_att, sr=self.sr, hop_length = STEP))
        for j in range(self.n_frames):
            for i in range(self.Nf):
                self.chromSync[i,j] = np.mean(self.Chrom[i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])

        #Normalisation du spectre
    #    ChromSync[:,1] = librosa.power_to_db(librosa.db_to_power(ChromSync[:,1]) / np.sum(librosa.db_to_power(ChromSync[:,1])))
        if norm_spectre:
            for j in range(self.n_frames):
                self.chromSync[:,j] = librosa.power_to_db(librosa.db_to_power(self.chromSync[:,j]) / np.sum(librosa.db_to_power(self.chromSync[:,j])))


        #Affichage
        if plot_onsets:
            plt.figure(figsize=(13, 7))
            ax1 = plt.subplot(3, 1, 1)
            librosa.display.specshow(self.Chrom, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=times)
            plt.title('CQT spectrogram')

            plt.subplot(3, 1, 2, sharex=ax1)
            plt.plot(times, Dev, label='Deviation')
            plt.plot(times, Seuil, color='g', label='Seuil')
            plt.vlines(times[Onsets], 0, Dev.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
            plt.axis('tight')
            plt.legend(frameon=True, framealpha=0.75)

            ax1 = plt.subplot(3, 1, 3, sharex=ax1)
            librosa.display.specshow(self.chromSync, bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time',x_coords=self.onset_times)
            plt.show()


    def clustering(self):
        """ Découpe et synchronise les pistes séparées sur les ONSET, stoque le spectrogramme
        synchronisé en construisant self.chromPistesSync"""

        #  Construction de chromPistesSync
        ChromPistes = []
        for k, voice in enumerate(self.pistes):
            ChromPistes.append(librosa.amplitude_to_db(np.abs(librosa.cqt(y=voice, sr=self.sr, hop_length = STEP, fmin= self.fmin, bins_per_octave=BINS_PER_OCTAVE, n_bins=self.n_bins)), ref=np.max))
            self.chromPistesSync.append(np.zeros((self.Nf,self.n_frames)))
            for j in range(self.n_frames):
                for i in range(self.Nf):
                    self.chromPistesSync[k][i,j] = np.mean(ChromPistes[k][i][(self.onset_frames[j]+self.n_att):(self.onset_frames[j+1]-self.n_att)])
        #  Plot
        if plot_pistes:
            plt.figure(figsize=(13, 7))
            ax1 = plt.subplot(self.n_pistes,1,1)
            librosa.display.specshow(self.chromPistesSync[0], bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            for k in range(1, self.n_pistes):
                plt.subplot(self.n_pistes, 1, k+1, sharex=ax1)
                librosa.display.specshow(self.chromPistesSync[k], bins_per_octave=BINS_PER_OCTAVE, fmin=self.fmin, y_axis='cqt_note', x_axis='time', x_coords=self.onset_times)
            plt.show()




    def concordance(self):
        self.chromConc = librosa.(db_to_power(self.chromConc))
        self.chromConc = np.ones((self.Nf,self.n_frames)))
        for k in range(n_pistes):
            self.chromConc = np.multiply(self.chromConc, chromPistesSync[k])






title = 'Palestrina'
#Palestrina, Cadence4VMaj
y, sr = librosa.load('/Users/manuel/Github/DescripteursHarmoniquesAudio/'+title+'.wav')
Notemin = 'D3'
Notemax = 'D9'

S = SignalSepare(y, sr, [y,y,y], Notemin, Notemax)
S.detectionOnsets()
S.clustering()
